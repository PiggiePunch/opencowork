import Anthropic from '@anthropic-ai/sdk';
import { SummaryRequest, SummaryResult } from './types';
import logger from '../../services/Logger';

/**
 * AI 摘要生成器
 * 复用 AgentRuntime 的 Anthropic 实例，确保使用相同的配置
 */
export class Summarizer {
    private anthropic: Anthropic;

    constructor(
        apiKey: string,
        apiUrl: string,
        private model: string
    ) {
        // 使用传入的配置创建 Anthropic 实例
        this.anthropic = new Anthropic({
            apiKey,
            baseURL: apiUrl
        });
        logger.info(`[Summarizer] Initialized with model: ${model}, apiUrl: ${apiUrl}`);
    }

    /**
     * 生成对话摘要
     * 参考 RooCode 提示词结构
     */
    async summarize(request: SummaryRequest): Promise<SummaryResult> {
        const startTime = Date.now();

        try {
            logger.info(`[Summarizer] Starting conversation summary using model: ${this.model}...`);

            // 1. 提取需要保留的工具结果
            const preservedContext: any[] = request.preserveToolResults
                ? this.extractToolResults(request.messages)
                : [];

            // 2. 构建摘要提示词
            const summaryPrompt = this.buildSummaryPrompt(request.messages);

            // 3. 调用 AI 生成摘要（使用与对话相同的方式）
            const summary = await this.callSummaryAPI(
                summaryPrompt,
                request.maxTokens
            );

            const duration = Date.now() - startTime;
            logger.info(`[Summarizer] Summary completed in ${duration}ms using ${this.model}, length: ${summary.length} chars`);

            return {
                summary,
                preservedContext,
                success: true
            };
        } catch (error: any) {
            const duration = Date.now() - startTime;
            logger.error(`[Summarizer] Summary failed after ${duration}ms using ${this.model}:`, error);

            return {
                summary: '',
                preservedContext: [],
                success: false,
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * 构建摘要提示词
     * 参考 RooCode 的提示词结构
     */
    private buildSummaryPrompt(messages: any[]): string {
        // 提取对话内容
        const conversationText = this.extractConversationText(messages);

        return `<conversation>
${conversationText}
</conversation>

Please summarize the above conversation concisely. Focus on:

1. **Previous Conversation**: What was discussed before the current work
2. **Current Work**: What is being worked on right now
3. **Key Technical Concepts**: Important technical details discussed
4. **Relevant Files and Code**: Files that were read or modified
5. **Problem Solving**: Issues encountered and how they were resolved
6. **Pending Tasks and Next Steps**: What still needs to be done

Format your response as a structured summary that can be used to restore context efficiently. Be concise but preserve important details.

Provide the summary in:`;
    }

    /**
     * 从消息中提取纯文本对话
     */
    private extractConversationText(messages: any[]): string {
        const parts: string[] = [];

        for (const msg of messages) {
            const role = msg.role === 'user' ? 'User' : 'Assistant';
            const content = this.extractTextFromMessage(msg);

            if (content && content.trim()) {
                // 限制每条消息的长度，避免摘要过长
                const truncatedContent = content.length > 2000
                    ? content.substring(0, 2000) + '...[truncated]'
                    : content;

                parts.push(`${role}: ${truncatedContent}`);
            }
        }

        return parts.join('\n\n');
    }

    /**
     * 从消息中提取纯文本内容
     */
    private extractTextFromMessage(message: any): string | null {
        if (typeof message.content === 'string') {
            return message.content;
        } else if (Array.isArray(message.content)) {
            const textBlocks = message.content
                .filter((block: any) => block.type === 'text')
                .map((block: any) => block.text || '')
                .filter((text: string) => text.trim());

            return textBlocks.join('\n\n') || null;
        }
        return null;
    }

    /**
     * 提取需要保留的工具结果
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
     * 调用 AI API 生成摘要
     */
    private async callSummaryAPI(
        conversation: string,
        maxTokens: number
    ): Promise<string> {
        try {
            logger.debug('[Summarizer] Calling summary API...');

            const response = await this.anthropic.messages.create({
                model: this.model,
                max_tokens: maxTokens,
                messages: [
                    {
                        role: 'user',
                        content: conversation
                    }
                ]
            });

            // 提取摘要文本
            let summary = '';
            if (response.content && response.content.length > 0) {
                const textBlocks = response.content.filter(
                    (block: any) => block.type === 'text'
                );

                summary = textBlocks
                    .map((block: any) => block.text || '')
                    .join('\n\n')
                    .trim();
            }

            if (!summary) {
                throw new Error('Empty summary generated');
            }

            logger.debug('[Summarizer] Summary generated successfully');
            return summary;
        } catch (error: any) {
            logger.error('[Summarizer] API call failed:', error);
            throw error;
        }
    }
}
