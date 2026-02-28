/**
 * 压缩模块通用消息类型
 * 不依赖任何特定 SDK，兼容所有使用 Anthropic 协议的 API
 */

export type MessageRole = 'user' | 'assistant' | 'system';

export type TextBlock = {
    type: 'text';
    text: string;
    citations?: any;
};

export type ImageBlock = {
    type: 'image';
    source: {
        type: 'base64';
        media_type: string;
        data: string;
    };
};

export type ToolUseBlock = {
    type: 'tool_use';
    id: string;
    name: string;
    input: any;
};

export type ToolResultBlock = {
    type: 'tool_result';
    tool_use_id: string;
    content: string | any[];
    is_error?: boolean;
};

export type ThinkingBlock = {
    type: 'thinking';
    text: string;
};

export type ContentBlock = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock;

export interface MessageParam {
    role: MessageRole;
    content: string | ContentBlock[];
}

// 为了兼容性，导出为 any 类型的别名
export type AnyMessageParam = any;

/**
 * 压缩策略枚举
 */
export enum CompressionStrategy {
    NONE = 'none',                         // 不压缩
    SMART_SUMMARY = 'smart_summary',       // 智能摘要
    TRUNCATE = 'truncate',                 // 滑动窗口截断
    AGGRESSIVE = 'aggressive'              // 激进压缩（组合策略）
}

/**
 * 压缩结果接口
 */
export interface CompressionResult {
    strategy: CompressionStrategy;
    originalMessages: any[];              // 兼容不同来源的消息类型
    compressedMessages: any[];            // 兼容不同来源的消息类型
    originalTokens: number;
    compressedTokens: number;
    compressionRatio: number;              // 压缩比例 (0-1)
    summary?: string;                      // 摘要内容（如果有）
    metadata: CompressionMetadata;
}

/**
 * 压缩元数据
 */
export interface CompressionMetadata {
    timestamp: number;
    duration: number;                      // 压缩耗时（ms）
    triggerReason: string;                 // 触发原因
    success: boolean;
    error?: string;
    messagesRemoved: number;
    toolResultsPreserved: number;
}

/**
 * 压缩触发条件
 */
export interface CompressionTrigger {
    currentTokens: number;
    maxTokens: number;
    thresholdPercent: number;
    shouldCompress: boolean;
    reason?: string;
}

/**
 * 摘要请求参数
 */
export interface SummaryRequest {
    messages: any[];                      // 兼容不同来源的消息类型
    maxTokens: number;
    preserveToolResults: boolean;
    customPrompt?: string;
}

/**
 * 摘要结果
 */
export interface SummaryResult {
    summary: string;
    preservedContext: any[];              // 兼容不同来源的消息类型
    success: boolean;
    error?: string;
}

/**
 * 压缩配置接口
 */
export interface CompressionConfig {
    // 基础配置
    enabled: boolean;                      // 是否启用智能压缩
    maxContextSize: number;                // 最大上下文大小（tokens）
    compressionThreshold: number;          // 触发压缩的阈值（百分比 0-100）

    // 智能摘要配置
    autoCondenseEnabled: boolean;          // 是否启用 AI 摘要压缩
    condenseThresholdPercent: number;      // 触发摘要的百分比（默认 60）
    summaryModel?: string;                 // 摘要使用的模型（默认使用当前模型）
    maxSummaryTokens: number;              // 摘要最大 tokens（默认 8000）

    // 截断回退配置
    truncateFallbackEnabled: boolean;      // 是否启用截断回退
    truncateKeepMessages: number;          // 截断时保留的消息数量
    truncateKeepToolResults: number;       // 截断时保留的工具结果数量

    // 高级配置
    preservePatterns: string[];            // 保留的消息模式（正则表达式）
    aggressiveMode: boolean;               // 激进模式（更早触发压缩）
}
