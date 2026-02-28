import { Summarizer } from './Summarizer';
import {
    CompressionStrategy,
    CompressionConfig,
    CompressionTrigger,
    CompressionResult,
    CompressionMetadata,
    MessageParam
} from './types';
import logger from '../../services/Logger';

/**
 * 上下文压缩器
 * 参考 RooCode 的压缩策略，实现多级压缩机制
 */
export class ContextCompressor {
    constructor(
        private config: CompressionConfig,
        private summarizer: Summarizer
    ) {}

    /**
     * 检查是否需要压缩
     */
    shouldCompress(messages: any[]): CompressionTrigger {
        const currentTokens = this.estimateTokens(messages);
        const maxTokens = this.config.maxContextSize;
        const thresholdPercent = this.config.aggressiveMode
            ? this.config.condenseThresholdPercent
            : this.config.compressionThreshold;

        const currentPercent = (currentTokens / maxTokens) * 100;
        const shouldCompress = currentPercent >= thresholdPercent;

        return {
            currentTokens,
            maxTokens,
            thresholdPercent,
            shouldCompress,
            reason: shouldCompress
                ? `Context usage (${currentPercent.toFixed(1)}%) exceeds threshold (${thresholdPercent}%)`
                : undefined
        };
    }

    /**
     * 主压缩入口：根据配置自动选择最佳压缩策略
     */
    async compress(
        messages: any[],
        trigger: CompressionTrigger
    ): Promise<CompressionResult> {
        const startTime = Date.now();
        const originalTokens = this.estimateTokens(messages);

        logger.info(`
╔══════════════════════════════════════════════════════════════╗
║              Context Compression Started                    ║
╠══════════════════════════════════════════════════════════════╣
║ Trigger Reason:     ${trigger.reason?.padEnd(46) || 'N/A'}║
║ Current Tokens:     ${trigger.currentTokens.toString().padEnd(46)}║
║ Max Tokens:         ${trigger.maxTokens.toString().padEnd(46)}║
║ Threshold:          ${trigger.thresholdPercent.toString() + '%'.padEnd(46)}║
║ Strategy:           Determining...                           ║
╚══════════════════════════════════════════════════════════════╝
        `);

        let result: CompressionResult;

        // 策略选择逻辑
        if (this.config.autoCondenseEnabled) {
            // 优先尝试智能摘要
            result = await this.compressWithSummary(messages);
        } else if (this.config.truncateFallbackEnabled) {
            // 直接使用截断
            result = await this.compressWithTruncate(messages);
        } else {
            // 没有启用任何压缩，返回原消息
            result = this.createNoCompressionResult(messages);
        }

        // 如果摘要失败，回退到截断
        if (!result.metadata.success && this.config.truncateFallbackEnabled) {
            logger.warn('[ContextCompressor] Smart summary failed, falling back to truncation');
            result = await this.compressWithTruncate(messages);
        }

        // 计算最终统计
        const duration = Date.now() - startTime;
        result.originalTokens = originalTokens;
        result.compressedTokens = this.estimateTokens(result.compressedMessages);
        result.compressionRatio = 1 - (result.compressedTokens / result.originalTokens);
        result.metadata.duration = duration;
        result.metadata.timestamp = Date.now();

        // 记录压缩结果日志
        this.logCompressionResult(result);

        return result;
    }

    /**
     * 智能摘要压缩（参考 RooCode）
     */
    private async compressWithSummary(
        messages: any[]
    ): Promise<CompressionResult> {
        const startTime = Date.now();

        try {
            logger.info('[ContextCompressor] Using SMART_SUMMARY strategy');

            // 1. 保留最近的对话（保持连贯性）
            const RECENT_MESSAGES_TO_KEEP = 10;
            const recentMessages = messages.slice(-RECENT_MESSAGES_TO_KEEP);
            const olderMessages = messages.slice(0, -RECENT_MESSAGES_TO_KEEP);

            // 2. 生成摘要
            const summaryResult = await this.summarizer.summarize({
                messages: olderMessages,
                maxTokens: this.config.maxSummaryTokens,
                preserveToolResults: true
            });

            if (!summaryResult.success) {
                throw new Error(summaryResult.error || 'Summary generation failed');
            }

            // 3. 构建压缩后的消息数组
            const compressedMessages: MessageParam[] = [];

            // 添加摘要作为系统消息
            if (summaryResult.summary) {
                compressedMessages.push({
                    role: 'user',
                    content: `[Previous Conversation Summary]\n\n${summaryResult.summary}\n\n---\n\n*This summary was generated to reduce context size while preserving important information.*`
                });
            }

            // 添加保留的工具结果
            if (summaryResult.preservedContext.length > 0) {
                compressedMessages.push(...summaryResult.preservedContext);
            }

            // 添加最近的对话
            compressedMessages.push(...recentMessages);

            const duration = Date.now() - startTime;
            const metadata: CompressionMetadata = {
                timestamp: Date.now(),
                duration,
                triggerReason: 'Auto-condense triggered',
                success: true,
                messagesRemoved: messages.length - compressedMessages.length,
                toolResultsPreserved: summaryResult.preservedContext.length
            };

            return {
                strategy: CompressionStrategy.SMART_SUMMARY,
                originalMessages: messages,
                compressedMessages,
                originalTokens: 0, // 会在 compress() 中计算
                compressedTokens: 0, // 会在 compress() 中计算
                compressionRatio: 0, // 会在 compress() 中计算
                summary: summaryResult.summary,
                metadata
            };
        } catch (error: any) {
            const duration = Date.now() - startTime;
            logger.error('[ContextCompressor] Smart summary failed:', error);

            // 返回失败结果，让上层决定是否回退
            const metadata: CompressionMetadata = {
                timestamp: Date.now(),
                duration,
                triggerReason: 'Auto-condense attempted',
                success: false,
                error: error.message || 'Unknown error',
                messagesRemoved: 0,
                toolResultsPreserved: 0
            };

            return {
                strategy: CompressionStrategy.SMART_SUMMARY,
                originalMessages: messages,
                compressedMessages: messages, // 失败时返回原消息
                originalTokens: 0,
                compressedTokens: 0,
                compressionRatio: 0,
                metadata
            };
        }
    }

    /**
     * 滑动窗口截断（回退策略）
     */
    private async compressWithTruncate(
        messages: any[]
    ): Promise<CompressionResult> {
        const startTime = Date.now();

        logger.info('[ContextCompressor] Using TRUNCATE strategy');

        try {
            // 1. 保留最近的消息
            const keepMessages = Math.min(
                this.config.truncateKeepMessages,
                messages.length
            );
            const recentMessages = messages.slice(-keepMessages);
            const removedMessages = messages.slice(0, -keepMessages);

            // 2. 从被移除的消息中提取工具结果
            const toolResults = this.extractToolResults(removedMessages);
            const preservedToolResults = toolResults.slice(
                -this.config.truncateKeepToolResults
            );

            // 3. 构建压缩后的消息数组
            const compressedMessages: MessageParam[] = [];

            // 添加截断提示
            if (removedMessages.length > 0) {
                compressedMessages.push({
                    role: 'user',
                    content: `[Context Truncated]\n\nTo stay within token limits, ${removedMessages.length} older messages were removed from the conversation history. The most recent ${keepMessages} messages have been preserved.`
                });
            }

            // 添加保留的工具结果
            compressedMessages.push(...preservedToolResults);

            // 添加最近的消息
            compressedMessages.push(...recentMessages);

            const duration = Date.now() - startTime;
            const metadata: CompressionMetadata = {
                timestamp: Date.now(),
                duration,
                triggerReason: 'Truncation triggered',
                success: true,
                messagesRemoved: removedMessages.length,
                toolResultsPreserved: preservedToolResults.length
            };

            return {
                strategy: CompressionStrategy.TRUNCATE,
                originalMessages: messages,
                compressedMessages,
                originalTokens: 0,
                compressedTokens: 0,
                compressionRatio: 0,
                metadata
            };
        } catch (error: any) {
            logger.error('[ContextCompressor] Truncation failed:', error);

            const metadata: CompressionMetadata = {
                timestamp: Date.now(),
                duration: Date.now() - startTime,
                triggerReason: 'Truncation attempted',
                success: false,
                error: error.message || 'Unknown error',
                messagesRemoved: 0,
                toolResultsPreserved: 0
            };

            return {
                strategy: CompressionStrategy.TRUNCATE,
                originalMessages: messages,
                compressedMessages: messages,
                originalTokens: 0,
                compressedTokens: 0,
                compressionRatio: 0,
                metadata
            };
        }
    }

    /**
     * 创建不压缩的结果
     */
    private createNoCompressionResult(
        messages: any[]
    ): CompressionResult {
        const metadata: CompressionMetadata = {
            timestamp: Date.now(),
            duration: 0,
            triggerReason: 'Compression disabled',
            success: true,
            messagesRemoved: 0,
            toolResultsPreserved: 0
        };

        return {
            strategy: CompressionStrategy.NONE,
            originalMessages: messages,
            compressedMessages: messages,
            originalTokens: this.estimateTokens(messages),
            compressedTokens: this.estimateTokens(messages),
            compressionRatio: 0,
            metadata
        };
    }

    /**
     * 提取工具结果
     */
    private extractToolResults(messages: any[]): any[] {
        const toolResults: any[] = [];

        for (const msg of messages) {
            if (Array.isArray(msg.content)) {
                const toolResultBlocks = msg.content.filter(
                    (block: any) => block.type === 'tool_result'
                );

                if (toolResultBlocks.length > 0) {
                    toolResults.push({
                        role: 'user',
                        content: toolResultBlocks
                    });
                }
            }
        }

        return toolResults;
    }

    /**
     * 估算消息的 token 数量（复用 AgentRuntime 的逻辑）
     */
    private estimateTokens(message: any): number;
    private estimateTokens(messages: any[]): number;
    private estimateTokens(messages: any | any[]): number {
        const single = !Array.isArray(messages);
        const msgArray = single ? [messages as MessageParam] : (messages as MessageParam[]);

        let totalTokens = 0;

        for (const msg of msgArray) {
            let text = '';

            if (typeof msg.content === 'string') {
                text = msg.content;
            } else if (Array.isArray(msg.content)) {
                text = msg.content.map(block => {
                    if (block.type === 'text') return (block as any).text || '';
                    if (block.type === 'image') return '[IMAGE]';
                    if (block.type === 'tool_use') {
                        return JSON.stringify(block.input);
                    }
                    if (block.type === 'tool_result') {
                        return typeof block.content === 'string'
                            ? block.content
                            : JSON.stringify(block.content);
                    }
                    return '';
                }).join('\n');
            }

            // 粗略估算：1 token ≈ 4 字符（英文），中文字符约 2 倍
            const chineseChars = (text.match(/[\u4e00-\u9fa5]/g) || []).length;
            const otherChars = text.length - chineseChars;
            totalTokens += Math.ceil((chineseChars * 2 + otherChars) / 4);
        }

        return single ? totalTokens : totalTokens;
    }

    /**
     * 记录压缩结果日志
     */
    private logCompressionResult(result: CompressionResult): void {
        logger.info(`
╔══════════════════════════════════════════════════════════════╗
║              Context Compression Completed                   ║
╠══════════════════════════════════════════════════════════════╣
║ Strategy:           ${result.strategy.padEnd(46)}║
║ Success:            ${result.metadata.success ? 'Yes'.padEnd(44) : 'No'.padEnd(44)}║
║ Duration:           ${result.metadata.duration + 'ms'.padEnd(46)}║
╠══════════════════════════════════════════════════════════════╣
║ Before Compression:                                           ║
║   Messages:         ${result.originalMessages.length.toString().padEnd(46)}║
║   Tokens:           ${result.originalTokens.toString().padEnd(46)}║
╠══════════════════════════════════════════════════════════════╣
║ After Compression:                                            ║
║   Messages:         ${result.compressedMessages.length.toString().padEnd(46)}║
║   Tokens:           ${result.compressedTokens.toString().padEnd(46)}║
║   Saved:            ${result.originalTokens - result.compressedTokens} tokens (${(result.compressionRatio * 100).toFixed(1)}%)${' '.repeat(Math.max(0, 20 - (result.compressionRatio * 100).toFixed(1).length))}║
╠══════════════════════════════════════════════════════════════╣
║ Details:                                                      ║
║   Messages Removed:  ${result.metadata.messagesRemoved.toString().padEnd(44)}║
║   Tool Results Kept: ${result.metadata.toolResultsPreserved.toString().padEnd(44)}║
╚══════════════════════════════════════════════════════════════╝
        `);

        // ⚠️ 新增：压缩前后详细对比
        logger.info('========== BEFORE COMPRESSION (First 5 messages) ==========');
        result.originalMessages.slice(0, 5).forEach((msg, idx) => {
            const contentPreview = this.extractMessagePreview(msg);
            const tokens = this.estimateTokens(msg);
            logger.info(`[${idx}] ${msg.role} (${tokens} tokens): ${contentPreview}`);
        });
        if (result.originalMessages.length > 5) {
            logger.info(`... and ${result.originalMessages.length - 5} more messages`);
        }

        logger.info('\n========== AFTER COMPRESSION (All messages) ==========');
        result.compressedMessages.forEach((msg, idx) => {
            const contentPreview = this.extractMessagePreview(msg);
            const tokens = this.estimateTokens(msg);
            logger.info(`[${idx}] ${msg.role} (${tokens} tokens): ${contentPreview}`);
        });

        logger.info('\n========== COMPRESSION DETAILS ==========');
        logger.info(`Strategy: ${result.strategy}`);
        logger.info(`Messages removed: ${result.originalMessages.length - result.compressedMessages.length}`);
        logger.info(`Tokens saved: ${result.originalTokens - result.compressedTokens} (${(result.compressionRatio * 100).toFixed(1)}%)`);
        logger.info(`Duration: ${result.metadata.duration}ms`);

        if (result.summary) {
            logger.info('\n========== GENERATED SUMMARY ==========');
            logger.info(result.summary);
            logger.info('========== END SUMMARY ==========\n');
        }

        if (result.metadata.error) {
            logger.error('\n========== COMPRESSION ERROR ==========');
            logger.error(result.metadata.error);
            logger.error('========== END ERROR ==========\n');
        }

        logger.info('========== END COMPRESSION DETAILS ==========\n');
    }

    /**
     * 提取消息预览文本
     */
    private extractMessagePreview(msg: any): string {
        if (typeof msg.content === 'string') {
            return msg.content.substring(0, 200) + (msg.content.length > 200 ? '...' : '');
        } else if (Array.isArray(msg.content)) {
            const previews = msg.content.map((block: any) => {
                if (block.type === 'text') {
                    return block.text?.substring(0, 200) || '';
                } else if (block.type === 'tool_use') {
                    return `[Tool: ${block.name}]`;
                } else if (block.type === 'tool_result') {
                    return `[Tool Result for ${block.tool_use_id}]`;
                } else if (block.type === 'image') {
                    return '[Image]';
                }
                return `[${block.type}]`;
            }).filter(Boolean).join(' ');

            return previews.substring(0, 200) + (previews.length > 200 ? '...' : '');
        }
        return '';
    }
}
