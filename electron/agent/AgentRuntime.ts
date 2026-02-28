import Anthropic from '@anthropic-ai/sdk';
import { BrowserWindow } from 'electron';

import { FileSystemTools, ReadFileSchema, WriteFileSchema, ListDirSchema, RunCommandSchema } from './tools/FileSystemTools';
import { SDKTools, EditSchema, GlobSchema, GrepSchema, WebFetchSchema, WebSearchSchema, TodoWriteSchema, AskUserQuestionSchema } from './tools/SDKTools';
import { SkillManager } from './skills/SkillManager';
import { MCPClientService } from './mcp/MCPClientService';
import { AutoMemoryManager } from '../memory/AutoMemoryManager';
import { permissionManager } from './security/PermissionManager';
import { configStore } from '../config/ConfigStore';
import { backgroundTaskManager, BackgroundTask } from './BackgroundTaskManager';
import { ContextCompressor } from './compression/ContextCompressor';
import { Summarizer } from './compression/Summarizer';
import os from 'os';
import fs from 'fs';
import logger from '../services/Logger';

// Safe commands that can be auto-approved in standard/trust modes
const SAFE_COMMANDS = [
    'python', 'python3', 'node', 'npm', 'pip', 'pip3', 'git', 'ls', 'cat', 'head', 'tail',
    'grep', 'find', 'echo', 'pwd', 'cd', 'ls -la', 'ls -l', 'ls -a', 'tree', 'wc', 'sort',
    'uniq', 'diff', 'patch', 'tar', 'unzip', 'zip', 'gzip', 'gunzip', 'bunzip2',
    'curl', 'wget', 'ping', 'traceroute', 'netstat', 'ps', 'top', 'htop'
];

// Dangerous patterns that always require confirmation
const DANGEROUS_PATTERNS = [
    /rm\s+-rf\s+/i, /rm\s+-r\s+/i, /del\s+\/s\s+\/q/i, /rd\s+\/s\s+\/q/i,
    /format\s+/i, /mkfs/i, /dd\s+if=/i, /shred/i,
    />\s*\/?dev\/(null|sda|sdb)/i, /2>\s*&1\s*>\s*\/dev\/null/i,
    /chmod\s+777/i, /chmod\s+-R\s+777/i, /chown\s+-R/i
];

// Check if a command is considered safe
function isSafeCommand(command: string): boolean {
    const trimmedCmd = command.trim();
    const baseCmd = trimmedCmd.split(' ')[0].toLowerCase();

    // Check if base command is in safe list
    if (SAFE_COMMANDS.some(safe => baseCmd === safe || trimmedCmd.startsWith(safe + ' '))) {
        return true;
    }

    // Check for dangerous patterns
    if (DANGEROUS_PATTERNS.some(pattern => pattern.test(trimmedCmd))) {
        return false;
    }

    // Read-only git commands are safe
    if (/^git\s+(log|show|diff|status|branch|remote|ls-files)/i.test(trimmedCmd)) {
        return true;
    }

    // Python/node with script files are generally safe
    if ((baseCmd === 'python' || baseCmd === 'python3' || baseCmd === 'node') &&
        (trimmedCmd.endsWith('.py') || trimmedCmd.endsWith('.js') || trimmedCmd.endsWith('.ts'))) {
        return true;
    }

    return false;
}

// Check if a write operation is potentially dangerous (overwriting existing file)
function isDangerousWrite(path: string): boolean {
    try {
        return fs.existsSync(path);
    } catch {
        return false;
    }
}


export type AgentMessage = {
    role: 'user' | 'assistant';
    content: string | Anthropic.ContentBlock[];
    id?: string;
};

export class AgentRuntime {
    private anthropic: Anthropic;
    private history: Anthropic.MessageParam[] = [];
    private windows: BrowserWindow[] = [];
    private fsTools: FileSystemTools;
    private sdkTools: SDKTools;
    private skillManager: SkillManager;
    private mcpService: MCPClientService;
    private autoMemory: AutoMemoryManager;
    private memoryContext: string = '';
    private abortController: AbortController | null = null;
    private isProcessing = false;
    private pendingConfirmations: Map<string, { resolve: (approved: boolean) => void }> = new Map();
    private pendingQuestions: Map<string, { resolve: (answer: string) => void }> = new Map();
    private artifacts: { path: string; name: string; type: string }[] = [];

    private model: string;
    private maxTokens: number;
    private lastProcessTime: number = 0;

    // ⚠️ 保存 API 配置用于压缩器
    private apiKey: string;
    private apiUrl: string;

    // Session isolation - track which session this agent instance is serving
    private sessionId: string | null = null;

    // ⚠️ 新增：历史版本号，防止旧数据覆盖新数据
    private historyVersion: number = 0;

    // Background task support
    private _isBackgroundMode: boolean = false;
    private _backgroundTaskId: string | null = null;
    private _onProgressCallback?: (taskId: string, progress: number, message: string) => void;

    // Custom system prompt (for specialized agents like Memory Assistant)
    private customSystemPrompt: string | null = null;

    // ⚠️ 新增：上下文压缩模块
    private contextCompressor: ContextCompressor | null = null;

    // Background mode status check (used to avoid "unused variable" errors)
    private isBackgroundTask(): boolean {
        return this._isBackgroundMode || this._backgroundTaskId !== null || this._onProgressCallback !== undefined;
    }

    constructor(apiKey: string, window: BrowserWindow, model: string = 'claude-3-5-sonnet-20241022', apiUrl: string = 'https://api.anthropic.com', maxTokens: number = 131072) {
        this.apiKey = apiKey;
        this.apiUrl = apiUrl;
        this.anthropic = new Anthropic({ apiKey, baseURL: apiUrl });
        this.model = model;
        this.maxTokens = maxTokens;
        this.windows = [window];
        this.fsTools = new FileSystemTools();
        this.sdkTools = new SDKTools(permissionManager, this);
        this.skillManager = new SkillManager();
        this.mcpService = new MCPClientService();
        this.autoMemory = new AutoMemoryManager();
    }

    public async initialize() {
        logger.debug('Initializing AgentRuntime...');
        try {
            // Parallelize loading for faster startup
            await Promise.all([
                this.skillManager.loadSkills(),
                this.mcpService.loadClients()
            ]);

            // ⚠️ 初始化上下文压缩模块
            this.initializeContextCompressor();

            logger.debug('AgentRuntime initialized (Skills & MCP loaded)');
        } catch (error) {
            logger.error('Failed to initialize AgentRuntime:', error);
        }
    }

    /**
     * 初始化上下文压缩模块
     */
    private initializeContextCompressor() {
        try {
            const compressionConfig = configStore.getCompressionConfig();
            // ⚠️ 使用保存的配置参数创建 Summarizer
            const summarizer = new Summarizer(
                this.apiKey,      // 使用保存的 API Key
                this.apiUrl,      // 使用保存的 API URL
                this.model        // 使用保存的模型
            );
            this.contextCompressor = new ContextCompressor(compressionConfig, summarizer);
            logger.debug('ContextCompressor initialized with config:', {
                enabled: compressionConfig.enabled,
                maxContextSize: compressionConfig.maxContextSize,
                compressionThreshold: compressionConfig.compressionThreshold,
                autoCondenseEnabled: compressionConfig.autoCondenseEnabled,
                summaryModel: this.model,
                summaryApiUrl: this.apiUrl
            });
        } catch (error) {
            logger.error('Failed to initialize ContextCompressor:', error);
            this.contextCompressor = null;
        }
    }

    // Hot-Swap Configuration without reloading context
    public updateConfig(model: string, apiUrl?: string, apiKey?: string, maxTokens?: number, contextLength?: number) {
        if (this.model === model && !apiUrl && !apiKey && maxTokens === undefined && contextLength === undefined) return;

        this.model = model;
        if (maxTokens !== undefined) {
            this.maxTokens = maxTokens;
        }
        // Re-create Anthropic client if credentials change
        if (apiUrl || apiKey) {
            // 更新保存的配置
            if (apiKey) this.apiKey = apiKey;
            if (apiUrl) this.apiUrl = apiUrl;

            this.anthropic = new Anthropic({
                apiKey: this.apiKey,
                baseURL: this.apiUrl
            });

            // ⚠️ 重新初始化压缩器以使用新的配置
            this.initializeContextCompressor();
        } else if (contextLength !== undefined) {
            // ⚠️ contextLength 变化也需要重新初始化压缩器
            this.initializeContextCompressor();
        }
        logger.debug(`Hot-Swap: Model updated to ${model}, maxTokens: ${this.maxTokens}`);
    }

    public removeWindow(win: BrowserWindow) {
        this.windows = this.windows.filter(w => w !== win);
    }

    // Handle confirmation response
    public handleConfirmResponse(id: string, approved: boolean) {
        const pending = this.pendingConfirmations.get(id);
        if (pending) {
            pending.resolve(approved);
            this.pendingConfirmations.delete(id);
        }
    }

    // Clear history for new session
    public clearHistory() {
        // Abort any running task first to prevent stream leakage to new session
        if (this.isProcessing) {
            logger.debug('Aborting running task before clearing history');
            this.abort();
        }

        this.history = [];
        this.artifacts = [];
        this.sessionId = null;  // Reset session ID
        this.notifyUpdate();
    }

    // Set custom system prompt (for specialized agents like Memory Assistant)
    public setSystemPrompt(prompt: string | null) {
        this.customSystemPrompt = prompt;
        logger.debug('Custom system prompt ' + (prompt ? 'set' : 'cleared'));
    }

    /**
     * 估算消息的 token 数量（粗略估算，约 1 token ≈ 4 字符）
     * 这是一个简单的估算方法，不是精确计算
     */
    private estimateTokens(message: Anthropic.MessageParam): number {
        let text = '';
        if (typeof message.content === 'string') {
            text = message.content;
        } else if (Array.isArray(message.content)) {
            text = message.content.map(block => {
                if (block.type === 'text') return block.text || '';
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
        return Math.ceil((chineseChars * 2 + otherChars) / 4);
    }

    /**
     * 从消息中提取纯文本内容
     */
    private extractTextFromMessage(message: Anthropic.MessageParam): string | null {
        if (typeof message.content === 'string') {
            return message.content;
        } else if (Array.isArray(message.content)) {
            const textBlock = message.content.find(b => b.type === 'text');
            return textBlock?.text || null;
        }
        return null;
    }

    /**
     * 智能裁剪历史记录以适应模型的上下文窗口
     *
     * 策略：
     * 1. 保留最近的 N 个消息（确保对话连贯性）
     * 2. 保留重要的系统消息（tool_use、tool_result）
     * 3. 保留包含决策、偏好等关键信息的用户消息
     * 4. 移除冗余的中间对话
     *
     * @param messages 完整的历史消息
     * @param maxContextTokens 最大上下文 token 数（默认 200000）
     * @returns 裁剪后的消息数组
     */
    private trimHistoryToFitContext(
        messages: Anthropic.MessageParam[],
        maxContextTokens: number = 200000
    ): Anthropic.MessageParam[] {
        if (messages.length === 0) return messages;

        // 第一步：保留最近的连续对话（确保连贯性）
        const MIN_RECENT_MESSAGES = 20; // 至少保留最近 20 条消息
        const keepRecent: Anthropic.MessageParam[] = [];
        let recentTokens = 0;

        // 从最新消息开始往前数，保留 MIN_RECENT_MESSAGES 条
        for (let i = messages.length - 1; i >= Math.max(0, messages.length - MIN_RECENT_MESSAGES); i--) {
            const msgTokens = this.estimateTokens(messages[i]);
            recentTokens += msgTokens;
            keepRecent.unshift(messages[i]);
        }

        // 如果最近消息已经超过限制，只返回这些
        if (recentTokens >= maxContextTokens) {
            logger.debug(`⚠️ Recent ${keepRecent.length} messages already exceed limit (~${recentTokens} tokens)`);
            return keepRecent;
        }

        // 第二步：添加更早的消息，优先级高的先添加
        const remainingBudget = maxContextTokens - recentTokens;
        const olderMessages = messages.slice(0, messages.length - MIN_RECENT_MESSAGES);
        const prioritizedOlder: Array<{ message: Anthropic.MessageParam; priority: number; tokens: number }> = [];

        for (const msg of olderMessages) {
            const tokens = this.estimateTokens(msg);
            let priority = 0;

            // 高优先级：包含工具调用的消息
            if (Array.isArray(msg.content)) {
                const hasToolUse = msg.content.some(block => block.type === 'tool_use');
                const hasToolResult = msg.content.some(block => block.type === 'tool_result');
                if (hasToolUse || hasToolResult) {
                    priority += 10;
                }
            }

            // 高优先级：包含关键词的用户消息（决策、偏好、学习）
            if (msg.role === 'user') {
                const content = this.extractTextFromMessage(msg);
                if (content) {
                    // 决策关键词
                    if (/(?:决定|选择|decided|choose|使用|use|采用|adopt|应该|should)/i.test(content)) {
                        priority += 15;
                    }
                    // 偏好关键词
                    if (/(?:我喜欢|我偏好|i prefer|i like|风格|style|习惯|habit)/i.test(content)) {
                        priority += 12;
                    }
                    // 知识关键词
                    if (/(?:学到了|learned|发现|found|理解|understand)/i.test(content)) {
                        priority += 10;
                    }
                    // 错误/问题
                    if (/(?:错误|error|问题|problem|bug|失败|fail)/i.test(content)) {
                        priority += 8;
                    }
                }
            }

            // 中优先级：助手的回复（通常包含重要信息）
            if (msg.role === 'assistant') {
                priority += 5;
            }

            prioritizedOlder.push({ message: msg, priority, tokens });
        }

        // 按优先级排序，高优先级的先加入
        prioritizedOlder.sort((a, b) => b.priority - a.priority);

        // 第三步：在预算内添加高优先级的旧消息
        let addedTokens = 0;
        const selectedOlder: Anthropic.MessageParam[] = [];

        for (const { message, tokens } of prioritizedOlder) {
            if (addedTokens + tokens <= remainingBudget) {
                selectedOlder.push(message);
                addedTokens += tokens;
            }
        }

        // 按原始顺序重新排序
        const allMessages = [...selectedOlder, ...keepRecent];
        allMessages.sort((a, b) => {
            const aIndex = messages.indexOf(a);
            const bIndex = messages.indexOf(b);
            return aIndex - bIndex;
        });

        if (allMessages.length < messages.length) {
            const totalTokens = recentTokens + addedTokens;
            logger.debug(`Smart trim: ${messages.length} → ${allMessages.length} messages (~${totalTokens} tokens, saved ${messages.length - allMessages.length} messages)`);

            // ⚠️ 自动保存被移除的重要信息到记忆
            this.saveTrimmedMemories(messages, allMessages).catch(err => {
                logger.error('Failed to save trimmed memories:', err);
            });
        }

        return allMessages;
    }

    /**
     * 自动保存被裁剪掉的重要信息到记忆
     */
    private async saveTrimmedMemories(
        originalMessages: Anthropic.MessageParam[],
        trimmedMessages: Anthropic.MessageParam[]
    ): Promise<void> {
        try {
            // 找出被移除的消息
            const trimmedSet = new Set(trimmedMessages);
            const removedMessages = originalMessages.filter(msg => !trimmedSet.has(msg));

            if (removedMessages.length === 0) return;

            // 提取重要信息
            const importantContent: string[] = [];
            for (const msg of removedMessages) {
                if (msg.role === 'user') {
                    const content = this.extractTextFromMessage(msg);
                    if (content) {
                        // 检查是否包含重要关键词
                        if (this.isImportantContent(content)) {
                            importantContent.push(`**User**: ${content.substring(0, 500)}`);
                        }
                    }
                } else if (msg.role === 'assistant') {
                    const content = this.extractTextFromMessage(msg);
                    if (content && content.length < 1000) {
                        // 只保存较短的助手回复（通常包含关键信息）
                        importantContent.push(`**Assistant**: ${content.substring(0, 500)}`);
                    }
                }
            }

            if (importantContent.length === 0) return;

            // 创建记忆内容
            const timestamp = new Date().toISOString().split('T')[0];
            const memoryContent = `
# Conversation Memory - ${timestamp}

## Trimmed Context
The following ${removedMessages.length} messages were trimmed from context to stay within token limits.

## Important Information

${importantContent.join('\n\n---\n\n')}

---
*Auto-saved at: ${new Date().toISOString()}*
`;

            // 保存到记忆文件
            await this.autoMemory.writeMemory(
                `conversation-history/${timestamp}.md`,
                memoryContent
            );

            logger.debug(`Auto-saved ${importantContent.length} important items to memory`);
        } catch (error) {
            logger.error('Failed to save trimmed memories:', error);
        }
    }

    /**
     * 判断内容是否重要（值得保存到记忆）
     */
    private isImportantContent(content: string): boolean {
        // 检查是否包含重要关键词
        const importantPatterns = [
            /(?:决定|选择|decided|choose|使用|use|采用|adopt|应该|should)/i, // 决策
            /(?:我喜欢|我偏好|i prefer|i like|风格|style|习惯|habit)/i, // 偏好
            /(?:学到了|learned|发现|found|理解|understand|记住|remember)/i, // 学习
            /(?:错误|error|问题|problem|bug|失败|fail|修复|fix)/i, // 问题/解决
            /(?:重要|important|关键|key|核心|core)/i, // 关键信息
            /(?:总结|summary|结论|conclusion)/i, // 总结
        ];

        return importantPatterns.some(pattern => pattern.test(content));
    }

    // Load history from saved session
    public loadHistory(messages: Anthropic.MessageParam[], sessionId?: string) {
        // Don't abort running task - let it continue
        // Just update the history and sessionId
        // The running task will continue to stream and the new history will be available after it completes

        const wasProcessing = this.isProcessing;

        this.history = messages;
        if (sessionId) {
            this.sessionId = sessionId;
            logger.debug(`Loaded history for session: ${sessionId} (wasProcessing: ${wasProcessing}, isProcessing: ${this.isProcessing})`);
        }
        this.artifacts = [];

        // Always notify update so frontend can see the loaded history
        // If processing, the streaming will continue and the history will be updated when done
        this.notifyUpdate();

        if (wasProcessing) {
            logger.debug(`Session ${sessionId} is processing, will continue streaming to frontend`);
        }
    }

    // Set the current session ID for this agent instance
    public setSessionId(sessionId: string) {
        this.sessionId = sessionId;
        logger.debug(`Set session ID: ${sessionId}`);
    }

    // Get the current session ID
    public getSessionId(): string | null {
        return this.sessionId;
    }

    // Add a window to the broadcast list (used by AgentManager)
    public addWindow(win: BrowserWindow): void {
        if (!this.windows.includes(win)) {
            this.windows.push(win);
            logger.debug(`Added window to session ${this.sessionId}. Total windows: ${this.windows.length}`);
        }
    }

    // Clean up destroyed windows from the broadcast list
    public cleanupDestroyedWindows(): void {
        const beforeCount = this.windows.length;
        this.windows = this.windows.filter(win => !win.isDestroyed());
        const afterCount = this.windows.length;

        if (beforeCount !== afterCount) {
            logger.debug(`Cleaned up ${beforeCount - afterCount} destroyed windows for session ${this.sessionId}. Remaining: ${afterCount}`);
        }
    }

    // Check if agent is currently processing
    public isProcessingMessage(): boolean {
        return this.isProcessing;
    }

    // Get last process time (for cleanup)
    public getLastProcessTime(): number {
        return this.lastProcessTime;
    }

    public async processUserMessage(input: string | { content: string, images: string[] }) {
        logger.debug(`processUserMessage called for session ${this.sessionId}`);

        // Prevent concurrent message processing
        if (this.isProcessing) {
            const timeSinceStart = Date.now() - this.lastProcessTime;

            // Auto-recover from stuck state if > 60 seconds have passed
            if (timeSinceStart > 60000) {
                logger.warn(`Detected stuck state for session ${this.sessionId} (${timeSinceStart}ms), auto-recovering`);
                this.isProcessing = false;
                this.abortController = null;
            } else {
                // Reject concurrent message within normal time window
                logger.warn(`Message blocked: session ${this.sessionId} is already processing (${timeSinceStart}ms ago)`);
                throw new Error('Cannot send message: another task is currently running. Please wait for it to complete or abort it first.');
            }
        }

        this.lastProcessTime = Date.now();
        this.isProcessing = true;
        this.abortController = new AbortController();

        try {
            await this.skillManager.loadSkills();
            await this.mcpService.loadClients();

            let userContent: string | Anthropic.ContentBlockParam[] = '';

            if (typeof input === 'string') {
                userContent = input;
            } else {
                const blocks: Anthropic.ContentBlockParam[] = [];
                // Process images
                if (input.images && input.images.length > 0) {
                    for (const img of input.images) {
                        // format: data:image/png;base64,......
                        const match = img.match(/^data:(image\/[a-zA-Z]+);base64,(.+)$/);
                        if (match) {
                            blocks.push({
                                type: 'image',
                                source: {
                                    type: 'base64',
                                    media_type: match[1] as 'image/jpeg' | 'image/png' | 'image/gif' | 'image/webp',
                                    data: match[2]
                                }
                            });
                        }
                    }
                }
                // Add text
                if (input.content && input.content.trim()) {
                    blocks.push({ type: 'text', text: input.content });
                } else if (blocks.some(b => b.type === 'image')) {
                    // [Fix] If only images are present, add a default prompt to satisfy API requirements
                    blocks.push({ type: 'text', text: "Please analyze this image." });
                }
                userContent = blocks;
            }

            // ⚠️ 自动检查相关记忆（静默，用户看不到）
            const userMessageText = typeof input === 'string' ? input : input.content;
            this.memoryContext = await this.autoMemory.checkRelevantMemories(userMessageText);

            if (this.memoryContext && this.memoryContext.trim()) {
                logger.debug(`Loaded relevant memories (${this.memoryContext.length} chars)`);
            }

            // Add user message to history
            this.history.push({ role: 'user', content: userContent });
            this.notifyUpdate();

            // Start the agent loop
            await this.runLoop();

            // ⚠️ 自动保存重要信息到记忆（后台，用户看不到）
            const lastAssistantMessage = this.history[this.history.length - 1];
            if (lastAssistantMessage && lastAssistantMessage.role === 'assistant') {
                const assistantResponse = JSON.stringify(lastAssistantMessage.content);
                this.autoMemory.analyzeAndSave(userMessageText, assistantResponse).catch(err => {
                    logger.error('[AgentRuntime] Failed to save to memory:', err);
                });
            }

        } catch (error: unknown) {
            const err = error as { status?: number; message?: string; error?: { message?: string; type?: string } };
            logger.error('Agent Loop Error:', error);

            // [Fix] Handle MiniMax/provider sensitive content errors gracefully
            if (err.status === 500 && (err.message?.includes('sensitive') || JSON.stringify(error).includes('1027'))) {
                this.broadcast('agent:error', 'AI Provider Error: The generated content was flagged as sensitive and blocked by the provider.');
            } else if (err.error?.type === 'invalid_request_error' && err.error?.message?.includes('tools[')) {
                // Tool name validation error - provide helpful message
                this.broadcast('agent:error', `配置错误: MCP 工具名称格式不正确\n\n详细信息: ${err.error.message}\n\n这通常是因为 MCP 服务器返回的工具名称包含了特殊字符（如中文）。请尝试：\n1. 禁用有问题的 MCP 服务器\n2. 或联系开发者修复此问题\n\n错误代码: ${err.status || 400}`);
            } else if (err.status === 400) {
                // Generic 400 error with details
                const details = err.error?.message || err.message || 'Unknown error';
                this.broadcast('agent:error', `请求错误 (400): ${details}\n\n请检查：\n- API Key 是否正确\n- API 地址是否有效\n- 模型名称是否正确`);
            } else if (err.status === 401) {
                this.broadcast('agent:error', `认证失败 (401): API Key 无效或已过期\n\n请检查您的 API Key 配置。`);
            } else if (err.status === 429) {
                this.broadcast('agent:error', `请求过多 (429): API 调用频率超限\n\n请稍后再试或升级您的 API 套餐。`);
            } else if (err.status === 500) {
                this.broadcast('agent:error', `服务器错误 (500): AI 服务提供商出现问题\n\n${err.message || '请稍后再试。'}`);
            } else if (err.status === 503) {
                this.broadcast('agent:error', `服务不可用 (503): AI 服务暂时无法访问\n\n请稍后再试或检查服务状态。`);
            } else {
                // Generic error with full details
                const errorMsg = err.message || err.error?.message || 'An unknown error occurred';
                const statusInfo = err.status ? `[${err.status}] ` : '';
                this.broadcast('agent:error', `${statusInfo}${errorMsg}`);
            }
        } finally {
            // 清空记忆上下文
            this.memoryContext = '';

            // Force reload MCP clients on next run if we had an error, to ensure fresh connection
            if (this.isProcessing && this.abortController?.signal.aborted) {
                // Was aborted, do nothing special
            } else {
                // For now, we don't force reload every time, but we ensure state is clear
            }

            this.isProcessing = false;
            this.abortController = null;
            logger.debug(`processUserMessage completed for session ${this.sessionId}`);
            this.notifyUpdate();
            // Broadcast done event to signal processing is complete
            this.broadcast('agent:done', {
                timestamp: Date.now(),
                sessionId: this.sessionId
            });

            // ⚠️ 新增：广播清空流式文本事件，确保前端显示完整历史
            this.broadcast('agent:clear-streaming', { sessionId: this.sessionId });

            // ✅ Auto-save history to SessionStore to prevent data loss
            // This ensures history is saved even if user switches to another session
            // ⚠️ 跳过记忆助手会话（它使用专用存储）
            const MEMORY_ASSISTANT_SESSION_ID = 'memory-assistant-session';
            if (this.sessionId && this.sessionId !== MEMORY_ASSISTANT_SESSION_ID && this.history.length > 0) {
                try {
                    const { sessionStoreV2 } = await import('../config/SessionStoreV2');
                    const hasRealContent = this.history.some(msg => {
                        const content = msg.content;
                        if (typeof content === 'string') {
                            return content.trim().length > 0;
                        } else if (Array.isArray(content)) {
                            return content.some(block =>
                                block.type === 'text' ? (block.text || '').trim().length > 0 : true
                            );
                        }
                        return false;
                    });

                    if (hasRealContent) {
                        sessionStoreV2.updateSessionImmediate(this.sessionId, this.history);
                        logger.debug(`✅ Auto-saved history to SessionStoreV2 for session ${this.sessionId}: ${this.history.length} messages`);
                    }
                } catch (error) {
                    logger.error(`❌ Error auto-saving session ${this.sessionId}:`, error);
                }
            } else if (this.sessionId === MEMORY_ASSISTANT_SESSION_ID) {
                // 记忆助手会话使用专用存储，在 main.ts 中处理
                logger.debug(`ℹ️ Skipping auto-save for memory assistant session (uses dedicated storage)`);
            }
        }
    }

    private async runLoop() {
        let keepGoing = true;
        let iterationCount = 0;
        const MAX_ITERATIONS = 30;

        while (keepGoing && iterationCount < MAX_ITERATIONS) {
            iterationCount++;
            logger.debug(`Loop iteration: ${iterationCount}`);
            if (this.abortController?.signal.aborted) break;

            const tools: Anthropic.Tool[] = [
                ReadFileSchema,
                WriteFileSchema,
                ListDirSchema,
                RunCommandSchema,
                EditSchema,
                GlobSchema,
                GrepSchema,
                WebFetchSchema,
                WebSearchSchema,
                TodoWriteSchema,
                AskUserQuestionSchema,
                ...(this.skillManager.getTools() as Anthropic.Tool[]),
                ...(await this.mcpService.getTools() as Anthropic.Tool[])
            ];

            // Build working directory context
            const authorizedFolders = permissionManager.getAuthorizedFolders();
            const workingDirContext = authorizedFolders.length > 0
                ? `\n\nWORKING DIRECTORY:\n- Primary: ${authorizedFolders[0]}\n- All authorized: ${authorizedFolders.join(', ')}\n\nYou should primarily work within these directories. Always use absolute paths.`
                : '\n\nNote: No working directory has been selected yet. Ask the user to select a folder first.';

            const skillsDir = os.homedir() + '/.opencowork/skills';

            // ⚠️ 自动检索相关记忆（仅在非记忆助手模式下）
            let memoryContext = '';
            if (!this.customSystemPrompt) {
                try {
                    // 从最近的用户消息中提取关键词用于检索
                    const recentUserMessages = this.history
                        .filter(m => m.role === 'user')
                        .slice(-3); // 只看最近 3 条用户消息

                    if (recentUserMessages.length > 0) {
                        const queryText = recentUserMessages
                            .map(m => this.extractTextFromMessage(m))
                            .filter(Boolean)
                            .join(' ')
                            .substring(0, 200); // 限制查询长度

                        if (queryText) {
                            memoryContext = await this.autoMemory.checkRelevantMemories(queryText);
                            if (memoryContext) {
                                logger.debug('[AgentRuntime] Found relevant memories, added to context');
                            }
                        }
                    }
                } catch (error) {
                    // 记忆检索失败不影响主流程
                    logger.error('[AgentRuntime] Memory retrieval failed:', error);
                }
            }

            // Use custom system prompt if set (for specialized agents like Memory Assistant)
            // Otherwise use the default OpenCowork system prompt with memory context
            const baseSystemPrompt = this.customSystemPrompt || `
# OpenCowork Assistant System

## Role Definition
You are OpenCowork, an advanced AI desktop assistant designed for efficient task execution, file management, coding assistance, and research. You operate in a secure local environment with controlled access to user-selected directories and specialized tools.

## Core Behavioral Principles

### Communication Style
- **Direct & Professional**: Be concise and purposeful. Avoid unnecessary pleasantries.
- **Execution-Focused**: Prioritize completing tasks efficiently over extensive discussion.
- **Proactive**: Verify tool availability before relying on them.

### Response Format
- Use Markdown for all structured content
- Prefer clear prose over bullet points for narrative content
- Use bullet points only for lists, summaries, or when explicitly requested

## Task Execution Guidelines

### Planning & Execution
**Internal Process (Not Visible to User):**
- Mentally break down complex requests into clear, actionable steps
- Identify required tools, dependencies, and potential obstacles
- Plan the most efficient execution path before starting

**External Output:**
- Start directly with execution or brief acknowledgment
- Provide natural progress updates during execution
- Focus on completed work, not planning intentions
- Use professional, results-oriented language

### File Management
- **Primary Workspace**: User-authorized directories (your main deliverable location)
- **Temporary Workspace**: System temp directories for intermediate processing
- **Security**: Never access files outside authorized directories without explicit permission

### Tool Usage Protocol
1. **Skills First**: Before any task, check for relevant skills in \`${skillsDir}\`
2. **MCP Integration**: Leverage available MCP servers for enhanced capabilities
3. **Tool Prefixes**: MCP tools use namespace prefixes (e.g., \`tool_name__action\`)

### ⚠️ Todo Management (Auto-Task Tracking)
**CRITICAL**: You MUST actively manage todos for all multi-step tasks:

1. **When to Create Todos**: Automatically create a todo list when:
   - User requests a task with 3+ steps
   - You identify a complex task requiring multiple actions
   - Starting a new feature implementation or debugging session

2. **Todo Format**:
   \`\`\`
   todo_write({
     "todos": [
       {"content": "Analyze current codebase", "activeForm": "Analyzing current codebase", "status": "in_progress"},
       {"content": "Implement feature X", "activeForm": "Implementing feature X", "status": "pending"},
       {"content": "Test and verify", "activeForm": "Testing and verifying", "status": "pending"}
     ]
   })
   \`\`\`

3. **Update Progress**:
   - Mark todo as \`in_progress\` when starting that task
   - Mark as \`completed\` when done
   - Add new todos if you discover additional steps
   - Call \`todo_write\` after EACH status change

4. **Benefits**:
   - User sees real-time progress
   - Better context for long-running tasks
   - Helps users understand what's happening

**Example**:
\`\`\`typescript
// Start task
todo_write({"todos": [
  {"activeForm": "Reading file", "content": "Read config file", "status": "in_progress"},
  {"activeForm": "Parsing data", "content": "Parse JSON data", "status": "pending"}
]})

// After first step completes
todo_write({"todos": [
  {"activeForm": "Reading file", "content": "Read config file", "status": "completed"},
  {"activeForm": "Parsing data", "content": "Parse JSON data", "status": "in_progress"}
]})
\`\`\`

## Current Context
**Working Directory**: ${workingDirContext}
**Skills Directory**: \`${skillsDir}\`

**Available Skills**:
${this.skillManager.getSkillMetadata().map(s => `- ${s.name}: ${s.description}`).join('\n')}

**Active MCP Servers**: ${JSON.stringify(this.mcpService.getActiveServers())}

${memoryContext ? `
**Relevant Memories**:
${memoryContext}
` : ''}

---
Remember: Plan internally, execute visibly. Focus on results, not process.`;

            // 组合最终系统提示
            const systemPrompt = baseSystemPrompt;

            logger.debug('Sending request to API...');
            logger.debug('Model:', this.model);
            logger.debug('Base URL:', this.anthropic.baseURL);

            // ⚠️ 默认尝试启用 Extended Thinking
            // 不管是什么模型(Claude、DeepSeek、MiniMax、智谱等),都尝试启用 thinking 参数
            // 如果 API 不支持,会返回错误,我们自动降级到普通模式
            // 这样可以兼容所有支持 thinking 的模型,不会被"全部打死"

            // ⚠️ 新增：智能上下文压缩（在 trimHistoryToFitContext 之前）
            let historyForRequest = this.history;
            if (this.contextCompressor) {
                try {
                    logger.debug(`[AgentRuntime] Checking if compression needed...`);
                    const compressionTrigger = this.contextCompressor.shouldCompress(this.history);

                    logger.debug(`[AgentRuntime] Compression check result:`, {
                        shouldCompress: compressionTrigger.shouldCompress,
                        currentTokens: compressionTrigger.currentTokens,
                        maxTokens: compressionTrigger.maxTokens,
                        thresholdPercent: compressionTrigger.thresholdPercent,
                        reason: compressionTrigger.reason
                    });

                    if (compressionTrigger.shouldCompress) {
                        logger.info(`[AgentRuntime] Context compression triggered: ${compressionTrigger.reason}`);

                        const compressionResult = await this.contextCompressor.compress(
                            this.history,
                            compressionTrigger
                        );

                        if (compressionResult.metadata.success) {
                            historyForRequest = compressionResult.compressedMessages;
                            logger.info(`[AgentRuntime] Compression successful! Messages: ${this.history.length} → ${historyForRequest.length}`);

                            // 通知前端压缩事件
                            this.broadcast('agent:context-compressed', {
                                strategy: compressionResult.strategy,
                                originalTokens: compressionResult.originalTokens,
                                compressedTokens: compressionResult.compressedTokens,
                                compressionRatio: compressionResult.compressionRatio
                            });
                        } else {
                            logger.warn('[AgentRuntime] Compression failed, using original history:', compressionResult.metadata.error);
                        }
                    } else {
                        logger.debug(`[AgentRuntime] No compression needed. Context usage: ${((compressionTrigger.currentTokens / compressionTrigger.maxTokens) * 100).toFixed(1)}%`);
                    }
                } catch (error) {
                    logger.error('[AgentRuntime] Compression error:', error);
                    // 压缩失败不影响主流程，继续使用原始历史
                }
            } else {
                logger.debug('[AgentRuntime] ContextCompressor not initialized, skipping compression check');
            }

            // ⚠️ 关键修复：裁剪历史记录以避免超过模型的上下文长度限制
            // 默认限制为 200k tokens，为 200k 上下文的模型留出安全余量
            const MAX_CONTEXT_TOKENS = 200000;
            const trimmedHistory = this.trimHistoryToFitContext(historyForRequest, MAX_CONTEXT_TOKENS);

            let stream: any;
            try {
                // ⚠️ 新增：记录请求日志
                this.logApiRequest(trimmedHistory, tools);

                const requestConfig: any = {
                    model: this.model,
                    max_tokens: this.maxTokens,
                    system: systemPrompt,
                    messages: trimmedHistory,  // 使用裁剪后的历史记录
                    stream: true,
                    tools: tools
                };

                // ✅ 默认尝试启用 thinking (所有模型)
                requestConfig.thinking = {
                    type: 'enabled',
                    budget_tokens: 20000  // 思考预算:最多 20000 tokens
                };
                logger.debug('[AgentRuntime] Attempting to enable Extended Thinking for model:', this.model);

                stream = await this.anthropic.messages.create(requestConfig, {
                    signal: this.abortController?.signal
                });

            } catch (thinkingError: any) {
                // 如果启用 thinking 导致错误,说明模型不支持,降级到普通模式
                if (thinkingError?.message?.includes('thinking') ||
                    thinkingError?.message?.includes('unsupported') ||
                    thinkingError?.message?.includes('unknown parameter') ||
                    thinkingError?.status === 400 ||
                    thinkingError?.status === 422) {
                    logger.warn('[AgentRuntime] Model does not support Extended Thinking, retrying without it:', thinkingError.message);

                    stream = await this.anthropic.messages.create({
                        model: this.model,
                        max_tokens: this.maxTokens,
                        system: systemPrompt,
                        messages: trimmedHistory,  // 使用裁剪后的历史记录
                        stream: true,
                        tools: tools
                    }, {
                        signal: this.abortController?.signal
                    });
                } else {
                    // 其他错误直接抛出
                    throw thinkingError;
                }
            }

            try {
                const finalContent: Anthropic.ContentBlock[] = [];
                let currentToolUse: { id: string; name: string; input: string } | null = null;
                let textBuffer = "";
                let thinkingBuffer = "";  // ⚠️ 新增:思考内容缓冲区

                // ⚠️ 关键修复：定期保存流式文本到 SessionStore，防止切换会话时丢失
                let tokenCount = 0;
                let lastSaveTime = Date.now();
                const SAVE_INTERVAL_TOKENS = 10; // 每 10 个 token 保存一次
                const SAVE_INTERVAL_MS = 1000; // 或每 1 秒保存一次

                // ⚠️ 关键修复：记录用户发送消息前的历史长度，用于正确替换流式内容
                const historyLengthBeforeStreaming = this.history.length;

                for await (const chunk of stream) {
                    if (this.abortController?.signal.aborted) {
                        stream.controller.abort();
                        break;
                    }

                    switch (chunk.type) {
                        case 'content_block_start':
                            if (chunk.content_block.type === 'tool_use') {
                                if (textBuffer) {
                                    finalContent.push({ type: 'text', text: textBuffer, citations: null });
                                    textBuffer = "";
                                }
                                currentToolUse = { ...chunk.content_block, input: "" };
                            } else if (chunk.content_block.type === 'thinking' || (chunk.content_block as any).type === 'reasoning') {
                                // ⚠️ Anthropic Extended Thinking 或 DeepSeek Reasoning
                                logger.debug('[AgentRuntime] Thinking block started');
                                thinkingBuffer = "";
                            }
                            break;
                        case 'content_block_delta':
                            if (chunk.delta.type === 'text_delta') {
                                textBuffer += chunk.delta.text;
                                tokenCount++;
                                // Broadcast streaming token to ALL windows
                                this.broadcast('agent:stream-token', chunk.delta.text);

                                // ⚠️ 定期保存流式内容到 SessionStore
                                const now = Date.now();
                                if (tokenCount % SAVE_INTERVAL_TOKENS === 0 || now - lastSaveTime > SAVE_INTERVAL_MS) {
                                    if (this.sessionId && textBuffer.length > 0) {
                                        try {
                                            const { sessionStoreV2 } = await import('../config/SessionStoreV2');
                                            // ⚠️ 关键修复：替换最后一条消息，而不是追加新消息
                                            // 构建临时消息历史：替换最后一条 assistant 消息为当前流式内容
                                            const partialMessage: Anthropic.MessageParam = {
                                                role: 'assistant',
                                                content: [{ type: 'text', text: textBuffer, citations: null }]
                                            };

                                            // 保留用户发送消息前的历史，替换/追加当前流式的部分响应
                                            const tempHistory = [
                                                ...this.history.slice(0, historyLengthBeforeStreaming),
                                                partialMessage
                                            ];

                                            // 使用立即保存，确保数据持久化
                                            sessionStoreV2.updateSessionImmediate(this.sessionId, tempHistory);
                                            logger.debug(`Auto-saved streaming content for session ${this.sessionId}: ${textBuffer.length} chars (replaced last message)`);
                                        } catch (error) {
                                            logger.error(`Error saving streaming content:`, error);
                                        }
                                        lastSaveTime = now;
                                    }
                                }
                            } else if (chunk.delta.type === 'thinking_delta' || (chunk.delta as any).type === 'reasoning_content' || (chunk.delta as any).reasoning) {
                                // ⚠️ Anthropic Extended Thinking 或其他推理模型的思考内容
                                const thinkingText = chunk.delta.thinking || (chunk.delta as any).text || (chunk.delta as any).reasoning || "";
                                thinkingBuffer += thinkingText;
                                // 实时广播思考内容给前端
                                this.broadcast('agent:stream-thinking', thinkingText);
                            } else if (chunk.delta.type === 'input_json_delta' && currentToolUse) {
                                currentToolUse.input += chunk.delta.partial_json;
                            }
                            break;
                        case 'content_block_stop':
                            if (currentToolUse) {
                                try {
                                    const parsedInput = JSON.parse(currentToolUse.input);
                                    finalContent.push({
                                        type: 'tool_use',
                                        id: currentToolUse.id,
                                        name: currentToolUse.name,
                                        input: parsedInput
                                    });
                                } catch (e) {
                                    logger.error("Failed to parse tool input", e);
                                    // Treat as a failed tool use so the model knows it messed up
                                    finalContent.push({
                                        type: 'tool_use',
                                        id: currentToolUse.id,
                                        name: currentToolUse.name,
                                        input: { error: "Invalid JSON input", raw: currentToolUse.input }
                                    });
                                }
                                currentToolUse = null;
                            } else if (thinkingBuffer) {
                                // ⚠️ Thinking block ended - 保存思考内容到 finalContent
                                logger.debug(`🧠 Thinking block ended: ${thinkingBuffer.length} chars, adding to finalContent`);
                                finalContent.push({ type: 'thinking', text: thinkingBuffer } as any);
                                thinkingBuffer = "";
                            } else if (textBuffer) {
                                // [Fix] Flush text buffer on block stop
                                logger.debug(`content_block_stop: flushing ${textBuffer.length} chars from textBuffer to finalContent`);
                                finalContent.push({ type: 'text', text: textBuffer, citations: null });
                                textBuffer = "";
                            }
                            break;
                        case 'message_stop':
                            logger.debug(`message_stop: textBuffer has ${textBuffer.length} chars`);
                            if (textBuffer) {
                                logger.debug(`message_stop: flushing ${textBuffer.length} chars from textBuffer to finalContent`);
                                finalContent.push({ type: 'text', text: textBuffer, citations: null });
                                textBuffer = "";
                            }
                            break;
                    }
                }

                // If aborted, save any partial content that was generated
                if (this.abortController?.signal.aborted) {
                    if (textBuffer) {
                        finalContent.push({ type: 'text', text: textBuffer + '\n\n[已中断]', citations: null });
                    }
                    if (finalContent.length > 0) {
                        const assistantMsg: Anthropic.MessageParam = { role: 'assistant', content: finalContent };
                        this.history.push(assistantMsg);
                        this.notifyUpdate();
                    }
                    // ⚠️ 关键修复: 不要 return,让 FINALLY 块执行以保存数据
                    // return; // ❌ 删除这行,让流程继续到 FINALLY 块
                }

                // ⚠️ 关键修复：确保 textBuffer 被完全处理
                // message_stop 后不应该还有内容,但为了安全起见,再次检查
                if (textBuffer && textBuffer.length > 0) {
                    logger.warn(`⚠️ textBuffer still has ${textBuffer.length} chars after stream loop, flushing to finalContent`);
                    finalContent.push({ type: 'text', text: textBuffer, citations: null });
                    textBuffer = "";
                }

                // ⚠️ 新增：记录响应日志
                this.logApiResponse(finalContent, textBuffer, thinkingBuffer);

                if (finalContent.length > 0) {
                    const assistantMsg: Anthropic.MessageParam = { role: 'assistant', content: finalContent };
                    this.history.push(assistantMsg);

                    // ✅ 立即保存完整消息到 SessionStore
                    if (this.sessionId) {
                        try {
                            const { sessionStoreV2 } = await import('../config/SessionStoreV2');
                            sessionStoreV2.updateSessionImmediate(this.sessionId, this.history);
                            logger.debug(`✅ Saved complete assistant message to SessionStoreV2 for session ${this.sessionId}: ${this.history.length} messages total`);
                        } catch (error) {
                            logger.error(`❌ Error saving complete message:`, error);
                        }
                    }

                    this.notifyUpdate();

                    const toolUses = finalContent.filter(c => c.type === 'tool_use');
                    if (toolUses.length > 0) {
                        const toolResults: Anthropic.ToolResultBlockParam[] = [];
                        for (const toolUse of toolUses) {
                            // Check abort before each tool execution
                            if (this.abortController?.signal.aborted) {
                                logger.debug('[AgentRuntime] Aborted before tool execution');
                                return;
                            }

                            if (toolUse.type !== 'tool_use') continue;

                            logger.debug(`Executing tool: ${toolUse.name}`);
                            let result = "Tool execution failed or unknown tool.";

                            try {
                                if (toolUse.name === 'read_file') {
                                    const args = toolUse.input as { path: string };
                                    if (!permissionManager.isPathAuthorized(args.path)) {
                                        result = `Error: Path ${args.path} is not in an authorized folder.`;
                                    } else {
                                        result = await this.fsTools.readFile(args);
                                    }
                                } else if (toolUse.name === 'write_file') {
                                    const args = toolUse.input as { path: string, content: string };
                                    if (!permissionManager.isPathAuthorized(args.path)) {
                                        result = `Error: Path ${args.path} is not in an authorized folder.`;
                                    } else {
                                        // Check trust level for write operations
                                        const trustLevel = configStore.getFileTrustLevel(args.path);
                                        const isNewFile = !isDangerousWrite(args.path);

                                        let approved = false;

                                        if (trustLevel === 'trust') {
                                            // Trust mode: auto-approve all writes
                                            approved = true;
                                        } else if (trustLevel === 'standard') {
                                            // Standard mode: first write needs confirmation, subsequent auto-approved
                                            // For simplicity: new files auto, existing files need confirm
                                            approved = isNewFile || await this.requestConfirmation(toolUse.name, `Write to file: ${args.path}`, args);
                                        } else {
                                            // Strict mode: always confirm
                                            approved = await this.requestConfirmation(toolUse.name, `Write to file: ${args.path}`, args);
                                        }

                                        if (approved) {
                                            result = await this.fsTools.writeFile(args);
                                            const fileName = args.path.split(/[\\/]/).pop() || 'file';
                                            this.artifacts.push({ path: args.path, name: fileName, type: 'file' });
                                            this.broadcast('agent:artifact-created', { path: args.path, name: fileName, type: 'file' });

                                            // ⚠️ 记录文件变更到 FileChangeTracker
                                            if (this.sessionId) {
                                                this.broadcast('file:record-change', {
                                                    filePath: args.path,
                                                    sessionId: this.sessionId,
                                                    toolUseId: toolUse.id
                                                });
                                            }
                                        } else {
                                            result = 'User denied the write operation.';
                                        }
                                    }
                                } else if (toolUse.name === 'list_dir') {
                                    const args = toolUse.input as { path: string };
                                    if (!permissionManager.isPathAuthorized(args.path)) {
                                        result = `Error: Path ${args.path} is not in an authorized folder.`;
                                    } else {
                                        result = await this.fsTools.listDir(args);
                                    }
                                } else if (toolUse.name === 'run_command') {
                                    const args = toolUse.input as { command: string, cwd?: string };
                                    const defaultCwd = authorizedFolders[0] || process.cwd();

                                    // Determine trust level from the working directory
                                    const trustLevel = args.cwd
                                        ? configStore.getFileTrustLevel(args.cwd)
                                        : configStore.getFileTrustLevel(defaultCwd);

                                    // Check if command is dangerous (always requires confirmation)
                                    const isDangerous = DANGEROUS_PATTERNS.some(pattern => pattern.test(args.command.trim()));
                                    if (isDangerous) {
                                        // Dangerous commands always need confirmation
                                        const approved = await this.requestConfirmation(toolUse.name, `Execute command: ${args.command}`, args);
                                        if (approved) {
                                            result = await this.fsTools.runCommand(args, defaultCwd);
                                        } else {
                                            result = 'User denied the command execution.';
                                        }
                                        return;
                                    }

                                    // Check if command is safe
                                    const isSafe = isSafeCommand(args.command);

                                    let approved = false;

                                    if (trustLevel === 'trust') {
                                        // Trust mode: auto-approve safe commands
                                        approved = true;
                                    } else if (trustLevel === 'standard') {
                                        // Standard mode: auto-approve safe commands
                                        approved = isSafe;
                                        if (!approved) {
                                            approved = await this.requestConfirmation(toolUse.name, `Execute command: ${args.command}`, args);
                                        }
                                    } else {
                                        // Strict mode: always confirm
                                        approved = await this.requestConfirmation(toolUse.name, `Execute command: ${args.command}`, args);
                                    }

                                    if (approved) {
                                        result = await this.fsTools.runCommand(args, defaultCwd);
                                    } else {
                                        result = 'User denied the command execution.';
                                    }
                                } else if (toolUse.name === 'edit_file') {
                                    const args = toolUse.input as { path: string; old_str: string; new_str: string; replace_all?: boolean };
                                    result = await this.sdkTools.editFile(args);
                                } else if (toolUse.name === 'glob') {
                                    const args = toolUse.input as { pattern: string; cwd?: string; includePattern?: string };
                                    result = await this.sdkTools.globFiles(args, authorizedFolders[0] || process.cwd());
                                } else if (toolUse.name === 'grep') {
                                    const args = toolUse.input as { pattern: string; path: string; glob?: string; caseInsensitive?: boolean; outputMode?: 'content' | 'files_with_matches' | 'count' };
                                    result = await this.sdkTools.grepContent(args);
                                } else if (toolUse.name === 'web_fetch') {
                                    const args = toolUse.input as { url: string; timeout?: number };
                                    result = await this.sdkTools.webFetch(args);
                                } else if (toolUse.name === 'web_search') {
                                    const args = toolUse.input as { query: string; numResults?: number };
                                    result = await this.sdkTools.webSearch(args);
                                } else if (toolUse.name === 'todo_write') {
                                    const args = toolUse.input as { todos: Array<{ content: string; activeForm: string; status: string }> };
                                    result = await this.sdkTools.todoWrite(args);
                                } else if (toolUse.name === 'ask_user_question') {
                                    const args = toolUse.input as { questions: Array<any> };
                                    result = await this.sdkTools.askUserQuestion(args);
                                } else {
                                    const skillInfo = this.skillManager.getSkillInfo(toolUse.name);
                                    logger.debug(`[Runtime] Skill ${toolUse.name} info found? ${!!skillInfo} (len: ${skillInfo?.instructions?.length})`);
                                    if (skillInfo) {
                                        // Return skill content following official Claude Code Skills pattern
                                        // The model should create scripts and run them from the skill directory
                                        result = `[SKILL LOADED: ${toolUse.name}]

SKILL DIRECTORY: ${skillInfo.skillDir}

Follow these instructions to complete the user's request. When the instructions reference Python modules in core/, create your script in the working directory and run it from the skill directory:

run_command: cd "${skillInfo.skillDir}" && python /path/to/your_script.py

Or add to the top of your script:
import sys; sys.path.insert(0, r"${skillInfo.skillDir}")

---
${skillInfo.instructions}
---`;
                                    } else if (toolUse.name.includes('__')) {
                                        result = await this.mcpService.callTool(toolUse.name, toolUse.input as Record<string, unknown>);
                                    } else if (toolUse.name.startsWith('mcp_')) {
                                        // Handle MCP Management Skill Tools
                                        const args = toolUse.input as any;
                                        if (toolUse.name === 'mcp_get_all_servers') {
                                            result = JSON.stringify(await this.mcpService.getAllServers(), null, 2);
                                        } else if (toolUse.name === 'mcp_add_server') {
                                            result = JSON.stringify(await this.mcpService.addServer(args.json_config), null, 2);
                                        } else if (toolUse.name === 'mcp_remove_server') {
                                            result = JSON.stringify(await this.mcpService.removeServer(args.name), null, 2);
                                        } else if (toolUse.name === 'mcp_toggle_server') {
                                            result = JSON.stringify(await this.mcpService.toggleServer(args.name, args.enabled), null, 2);
                                        } else if (toolUse.name === 'mcp_diagnose_server') {
                                            const status = (await this.mcpService.getAllServers()).find(s => s.name === args.name);
                                            if (status) {
                                                result = JSON.stringify(await this.mcpService.diagnoseServer(args.name, status.config), null, 2);
                                            } else {
                                                result = JSON.stringify({ success: false, message: "Server not found" });
                                            }
                                        } else if (toolUse.name === 'mcp_retry_connection') {
                                            // Force reconnect
                                            const status = (await this.mcpService.getAllServers()).find(s => s.name === args.name);
                                            if (status) {
                                                await this.mcpService['connectToServer'](status.name, status.config);
                                                result = `Retry initiated for ${args.name}`;
                                            } else {
                                                result = `Server ${args.name} not found`;
                                            }
                                        }
                                    }
                                }
                                // Check if input has parse error
                                const inputObj = toolUse.input as Record<string, unknown>;
                                if (inputObj && inputObj.error === "Invalid JSON input") {
                                    // Provide simpler error, just raw info
                                    result = `Error: The tool input was not valid JSON. Please fix the JSON format and retry. Raw input length: ${(inputObj.raw as string)?.length || 0}`;
                                }
                            } catch (toolErr: unknown) {
                                result = `Error executing tool: ${(toolErr as Error).message}`;
                            }

                            toolResults.push({
                                type: 'tool_result',
                                tool_use_id: toolUse.id,
                                content: result
                            });
                        }

                        this.history.push({ role: 'user', content: toolResults });
                        this.notifyUpdate();
                    } else {
                        keepGoing = false;
                    }
                } else {
                    keepGoing = false;
                }

            } catch (loopError: unknown) {
                // Check if this is an abort error - handle gracefully
                if (this.abortController?.signal.aborted) {
                    logger.debug('[AgentRuntime] Request was aborted');
                    return; // Exit cleanly on abort
                }

                const loopErr = loopError as { status?: number; message?: string; name?: string };
                logger.error("Agent Loop detailed error:", loopError);

                // Check for abort-related errors (different SDK versions may throw different errors)
                if (loopErr.name === 'AbortError' || loopErr.message?.includes('abort')) {
                    logger.debug('[AgentRuntime] Caught abort error');
                    return;
                }

                // Handle Sensitive Content Error (1027)
                if (loopErr.status === 500 && (loopErr.message?.includes('sensitive') || JSON.stringify(loopError).includes('1027'))) {
                    logger.debug("Caught sensitive content error, asking Agent to retry...");

                    // Add a system-like user message to prompt the agent to fix its output
                    this.history.push({
                        role: 'user',
                        content: `[SYSTEM ERROR] Your previous response was blocked by the safety filter (Error Code 1027: output new_sensitive). \n\nThis usually means the generated content contained sensitive, restricted, or unsafe material.\n\nPlease generate a NEW response that:\n1. Addresses the user's request safely.\n2. Avoids the sensitive topic or phrasing that triggered the block.\n3. Acknowledges the issue briefly if necessary.`
                    });
                    this.notifyUpdate();

                    // Allow the loop to continue to the next iteration
                    continue;
                } else {
                    // Re-throw other errors to be caught effectively by the outer handler
                    throw loopError;
                }
            }
        }
    }

    // Broadcast to all windows with session ID
    private broadcast(channel: string, data: unknown, options?: { version?: number }) {
        // Clean up destroyed windows before broadcasting
        this.cleanupDestroyedWindows();

        const eventData: any = {
            sessionId: this.sessionId,
            data: data
        };

        // ⚠️ 添加版本号（如果提供）
        if (options?.version !== undefined) {
            eventData.version = options.version;
        }

        // Log streaming events for debugging
        if (channel === 'agent:stream-token') {
            logger.debug(`Broadcasting stream-token for session ${this.sessionId}:`, typeof data === 'string' ? data.substring(0, 20) + '...' : data);
        }

        for (const win of this.windows) {
            if (!win.isDestroyed()) {
                win.webContents.send(channel, eventData);
            }
        }
    }

    private notifyUpdate() {
        this.broadcast('agent:history-update', this.history, { version: ++this.historyVersion });
    }

    private async requestConfirmation(tool: string, description: string, args: Record<string, unknown>): Promise<boolean> {
        // Extract path from args if available
        const path = (args?.path || args?.cwd) as string | undefined;

        // Check if permission is already granted
        if (configStore.hasPermission(tool, path)) {
            logger.debug(`Auto-approved ${tool} (saved permission)`);
            return true;
        }

        const id = `confirm-${Date.now()}-${Math.random().toString(36).substring(7)}`;
        return new Promise((resolve) => {
            this.pendingConfirmations.set(id, { resolve });
            this.broadcast('agent:confirm-request', { id, tool, description, args });
        });
    }

    public handleConfirmResponseWithRemember(id: string, approved: boolean, remember: boolean): void {
        const pending = this.pendingConfirmations.get(id);
        if (pending) {
            if (approved && remember) {
                // Extract tool and path from the confirmation request
                // The tool name is in the id or we need to pass it
                // For now we'll extract from the most recent confirm request
            }
            pending.resolve(approved);
            this.pendingConfirmations.delete(id);
        }
    }

    public abort() {
        if (!this.isProcessing) return;

        this.abortController?.abort();

        // Clear any pending confirmations - respond with 'denied'
        for (const [, pending] of this.pendingConfirmations) {
            pending.resolve(false);
        }
        this.pendingConfirmations.clear();

        // Clear any pending questions - respond with empty answers
        for (const [, pending] of this.pendingQuestions) {
            pending.resolve('');
        }
        this.pendingQuestions.clear();

        // Broadcast abort event to all windows
        this.broadcast('agent:aborted', {
            aborted: true,
            timestamp: Date.now()
        });

        // Mark processing as complete
        this.isProcessing = false;
        this.abortController = null;
    }

    /**
     * 添加待处理的用户问题
     */
    public addPendingQuestion(requestId: string, resolve: (answer: string) => void): void {
        this.pendingQuestions.set(requestId, { resolve });
    }

    /**
     * 处理用户对问题的回答
     */
    public handleUserQuestionAnswer(requestId: string, answers: string[]): void {
        const pending = this.pendingQuestions.get(requestId);
        if (pending && pending.resolve) {
            // 将答案数组格式化为字符串（每个答案一行）
            const answerText = answers.join('\n');
            pending.resolve(answerText);
            this.pendingQuestions.delete(requestId);
        }
    }

    /**
     * Set background mode for this runtime instance
     */
    public setBackgroundMode(
        isBackground: boolean,
        taskId?: string,
        onProgress?: (taskId: string, progress: number, message: string) => void
    ) {
        this._isBackgroundMode = isBackground;
        this._backgroundTaskId = taskId || null;
        this._onProgressCallback = onProgress;
    }

    /**
     * Process a message in background mode
     * Returns immediately and runs the task asynchronously
     */
    public async processInBackground(
        sessionId: string,
        taskTitle: string,
        messages: any[],
        apiKey: string,
        model: string,
        apiUrl: string,
        maxTokens: number
    ): Promise<string> {
        // Create a new background task
        const task = backgroundTaskManager.createTask(sessionId, taskTitle, messages);

        // Execute the task asynchronously (don't await)
        this.executeBackgroundTask(task, apiKey, model, apiUrl, maxTokens).catch((error) => {
            logger.error(`Background task failed:`, error);
            backgroundTaskManager.failTask(task.id, error.message);
        });

        // Return task ID immediately
        return task.id;
    }

    /**
     * Execute a background task
     */
    private async executeBackgroundTask(
        task: BackgroundTask,
        apiKey: string,
        model: string,
        apiUrl: string,
        maxTokens: number
    ) {
        // Update task status to running
        backgroundTaskManager.startTask(task.id);

        // Create a new runtime instance for this background task
        const backgroundRuntime = new AgentRuntime(
            apiKey,
            this.windows[0], // Use the same window
            model,
            apiUrl,
            maxTokens
        );

        // Set background mode
        backgroundRuntime.setBackgroundMode(
            true,
            task.id,
            (taskId, progress) => {
                backgroundTaskManager.updateTaskProgress(taskId, progress);
            }
        );

        // Initialize and load history
        await backgroundRuntime.initialize();
        backgroundRuntime.loadHistory(task.messages);

        try {
            // Process the message
            await backgroundRuntime['processUserMessage'](task.messages[task.messages.length - 1]);

            // Get the final history
            const finalHistory = backgroundRuntime['history'];
            const lastMessage = finalHistory[finalHistory.length - 1];

            // Extract result text
            let resultText = '';
            if (typeof lastMessage.content === 'string') {
                resultText = lastMessage.content;
            } else if (Array.isArray(lastMessage.content)) {
                resultText = lastMessage.content
                    .filter((block: any) => block.type === 'text')
                    .map((block: any) => block.text)
                    .join('\n');
            }

            // Mark task as completed
            backgroundTaskManager.completeTask(task.id, resultText);

            // Show notification (optional)
            this.broadcast('background-task:complete', {
                taskId: task.id,
                title: task.title,
                result: resultText,
            });
        } catch (error: any) {
            // Mark task as failed
            backgroundTaskManager.failTask(task.id, error.message || 'Unknown error');

            this.broadcast('background-task:failed', {
                taskId: task.id,
                title: task.title,
                error: error.message,
            });
        } finally {
            backgroundRuntime.dispose();
        }
    }

    /**
     * 记录 API 请求日志
     */
    private logApiRequest(messages: Anthropic.MessageParam[], tools: Anthropic.Tool[]) {
        const estimatedTokens = this.estimateHistoryTokens(messages);

        logger.info(`
╔══════════════════════════════════════════════════════════════╗
║                    API Request                                ║
╠══════════════════════════════════════════════════════════════╣
║ Model:              ${this.model.padEnd(46)}║
║ Max Tokens:         ${this.maxTokens.toString().padEnd(46)}║
║ History Messages:   ${messages.length.toString().padEnd(46)}║
║ Estimated Tokens:   ${estimatedTokens.toString().padEnd(46)}║
║ Tools Available:    ${tools.length.toString().padEnd(46)}║
╚══════════════════════════════════════════════════════════════╝
        `);

        // ⚠️ 记录完整的请求体
        logger.info('========== API REQUEST BODY ==========');
        logger.info(JSON.stringify({
            model: this.model,
            max_tokens: this.maxTokens,
            messages: messages.map(msg => ({
                role: msg.role,
                content: typeof msg.content === 'string'
                    ? msg.content.substring(0, 500) + (msg.content.length > 500 ? '...' : '')
                    : msg.content.map(block => {
                        if (block.type === 'text') {
                            return { type: 'text', text: (block as any).text?.substring(0, 500) + (((block as any).text?.length || 0) > 500 ? '...' : '') };
                        } else if (block.type === 'image') {
                            return { type: 'image' };
                        } else if (block.type === 'tool_use') {
                            return { type: 'tool_use', name: (block as any).name, input_length: JSON.stringify((block as any).input).length };
                        } else if (block.type === 'tool_result') {
                            return { type: 'tool_result', content_length: JSON.stringify((block as any).content).length };
                        }
                        return block;
                    })
            })),
            tools: tools.map(t => ({ name: t.name, description: t.description?.substring(0, 100) })),
            stream: true
        }, null, 2));
        logger.info('========== END REQUEST BODY ==========\n');
    }

    /**
     * 记录 API 响应日志
     */
    private logApiResponse(finalContent: Anthropic.ContentBlock[], textBuffer: string, thinkingBuffer: string) {
        const toolUses = finalContent.filter(c => c.type === 'tool_use');

        logger.info(`
╔══════════════════════════════════════════════════════════════╗
║                    API Response                               ║
╠══════════════════════════════════════════════════════════════╣
║ Content Blocks:     ${finalContent.length.toString().padEnd(46)}║
║ Tool Uses:          ${toolUses.length.toString().padEnd(46)}║
║ Text Length:        ${textBuffer.length.toString().padEnd(46)}║
║ Has Thinking:       ${(thinkingBuffer.length > 0 ? 'Yes' : 'No').padEnd(44)}║
╚══════════════════════════════════════════════════════════════╝
        `);

        // ⚠️ 记录完整的返回体
        logger.info('========== API RESPONSE BODY ==========');
        logger.info(JSON.stringify({
            content_blocks: finalContent.map(block => {
                if (block.type === 'text') {
                    return { type: 'text', text: (block as any).text?.substring(0, 1000) + (((block as any).text?.length || 0) > 1000 ? '...' : '') };
                } else if (block.type === 'tool_use') {
                    return {
                        type: 'tool_use',
                        id: (block as any).id,
                        name: (block as any).name,
                        input: JSON.stringify((block as any).input).substring(0, 500) + '...'
                    };
                } else if (block.type === 'thinking') {
                    return { type: 'thinking', length: (block as any).text?.length || 0 };
                }
                return { type: block.type };
            }),
            thinking_buffer_length: thinkingBuffer.length,
            thinking_buffer_preview: thinkingBuffer.substring(0, 500) + (thinkingBuffer.length > 500 ? '...' : ''),
            text_buffer_length: textBuffer.length,
            text_buffer_preview: textBuffer.substring(0, 500) + (textBuffer.length > 500 ? '...' : '')
        }, null, 2));
        logger.info('========== END RESPONSE BODY ==========\n');
    }

    /**
     * 估算历史记录的 tokens
     */
    private estimateHistoryTokens(history: Anthropic.MessageParam[]): number {
        return history.reduce((total, msg) => total + this.estimateTokens(msg), 0);
    }

    public dispose() {
        // Check if this was a background task (for cleanup/logging purposes)
        const wasBackgroundTask = this.isBackgroundTask();
        if (wasBackgroundTask) {
            logger.debug('[AgentRuntime] Disposing background task');
        }
        this.abort();
        this.mcpService.dispose();
    }
}
