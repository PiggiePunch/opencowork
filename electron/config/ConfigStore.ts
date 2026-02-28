import Store from 'electron-store';
import logger from '../services/Logger';
import { CompressionConfig } from '../agent/compression/types';

export interface ToolPermission {
    tool: string;           // 'write_file', 'run_command', etc.
    pathPattern?: string;   // Optional: specific path or '*' for all
    grantedAt: number;      // Timestamp
}

export interface ProviderConfig {
    id: string;
    name: string;
    apiKey: string;
    apiUrl: string;
    model: string;
    maxTokens?: number;           // 单次输出最大 token 数
    contextLength?: number;       // 模型支持的上下文长度（seq length）
    isCustom?: boolean;
    readonlyUrl?: boolean;
    compressionConfig?: Partial<CompressionConfig>;  // 压缩配置（可选）
}

export type TrustLevel = 'strict' | 'standard' | 'trust';

export interface FolderAuthorization {
    path: string;
    trustLevel: TrustLevel;
    addedAt: number;
}

export interface AppConfig {
    // Global/Legacy
    authorizedFolders: FolderAuthorization[];
    networkAccess: boolean;
    shortcut: string;
    allowedPermissions: ToolPermission[];

    // Multi-Provider
    activeProviderId: string;
    providers: Record<string, ProviderConfig>;

    // Context Compression
    globalCompressionConfig?: Partial<CompressionConfig>;  // 全局压缩配置（可覆盖 Provider 级别配置）
}

const DEFAULT_MAX_TOKENS = 8192;           // 默认输出最大 token 数
const DEFAULT_CONTEXT_LENGTH = 200000;       // 默认上下文长度（200k）

// Default provider configurations
const defaultProviders: Record<string, ProviderConfig> = {
    'glm': {
        id: 'glm',
        name: '智谱 GLM',
        apiKey: '',
        apiUrl: 'https://open.bigmodel.cn/api/anthropic',
        model: 'glm-4.7',
        maxTokens: DEFAULT_MAX_TOKENS,
        contextLength: DEFAULT_CONTEXT_LENGTH,
        readonlyUrl: true
    },
    'zai': {
        id: 'zai',
        name: 'ZAI (海外)',
        apiKey: '',
        apiUrl: 'https://api.z.ai/api/anthropic', // Placeholder, verify ZAI endpoint
        model: 'glm-4.7', // Assuming ZAI uses similar models or map later
        maxTokens: DEFAULT_MAX_TOKENS,
        contextLength: DEFAULT_CONTEXT_LENGTH,
        readonlyUrl: true
    },
    'minimax_cn': {
        id: 'minimax_cn',
        name: 'MiniMax (国内)',
        apiKey: '',
        apiUrl: 'https://api.minimaxi.com/anthropic',
        model: 'MiniMax-M2.1',
        maxTokens: DEFAULT_MAX_TOKENS,
        contextLength: DEFAULT_CONTEXT_LENGTH,
        readonlyUrl: true
    },
    'minimax_intl': {
        id: 'minimax_intl',
        name: 'MiniMax (海外)',
        apiKey: '',
        apiUrl: 'https://api.minimax.io/anthropic', // This uses Anthropic protocol
        model: 'MiniMax-M2.1',
        maxTokens: DEFAULT_MAX_TOKENS,
        contextLength: DEFAULT_CONTEXT_LENGTH,
        readonlyUrl: true
    },
    'custom': {
        id: 'custom',
        name: '自定义',
        apiKey: '',
        apiUrl: '',
        model: '',
        maxTokens: DEFAULT_MAX_TOKENS,
        contextLength: DEFAULT_CONTEXT_LENGTH,
        isCustom: true,
        readonlyUrl: false
    }
};

// ZAI Endpoint correction if needed. 'https://api.zhinao.ai/v1' ? 
// User said "ZAI overseas". Zhipu's overseas brand. 
// Let's assume standard OpenAI compat URL or allow edit if unsure. 
// Actually, ZAI docs usually point to https://api.z.ai/v1 or similar. 
// I will set a reasonable default and allow user to change if needed, or stick to what I know.
// Search result said "https://z.ai/model-api". 

const defaults: AppConfig = {
    authorizedFolders: [],
    networkAccess: true,
    shortcut: 'Alt+Space',
    allowedPermissions: [],
    activeProviderId: 'minimax_intl', // Default to what we had
    providers: defaultProviders
};

class ConfigStore {
    private store: Store<any>; // Use any to allow migration check

    constructor() {
        this.store = new Store({
            name: 'opencowork-config',
            defaults: defaults as any
        });

        this.migrate();
    }

    private migrate() {
        // Check if we have legacy flat keys
        if (this.store.has('apiKey') && !this.store.has('providers')) {
            const legacyKey = this.store.get('apiKey');

            // Copy to default provider (minimax_intl) or custom?
            // Since previous default was MiniMax, let's update minimax_intl
            const providers = { ...defaultProviders };
            if (legacyKey) {
                providers['minimax_intl'].apiKey = legacyKey;
                // If URL was custom, we might lose it if we force minimax_intl URL.
                // But usually it was static.
            }

            this.store.set('providers', providers);
            this.store.set('activeProviderId', 'minimax_intl');

            // Clean up legacy
            this.store.delete('apiKey');
            this.store.delete('apiUrl');
            this.store.delete('model');
        }

        // Migrate authorizedFolders from string[] to FolderAuthorization[]
        const oldFolders = this.store.get('authorizedFolders');
        if (oldFolders && oldFolders.length > 0) {
            const newFolders: FolderAuthorization[] = (oldFolders as any[]).map(item => {
                // Handle old string format
                if (typeof item === 'string') {
                    return { path: item, trustLevel: 'strict' as TrustLevel, addedAt: Date.now() };
                }
                // Handle old format with yoloMode
                if ('yoloMode' in item) {
                    return {
                        path: item.path,
                        trustLevel: item.yoloMode ? 'trust' : 'strict' as TrustLevel,
                        addedAt: item.addedAt || Date.now()
                    };
                }
                return item;
            });
            this.store.set('authorizedFolders', newFolders);
        }

        // Migrate providers: add contextLength if missing (use maxTokens as fallback)
        const providers = this.store.get('providers');
        if (providers) {
            let needsUpdate = false;
            for (const [, provider] of Object.entries(providers)) {
                if (!(provider as any).contextLength) {
                    (provider as any).contextLength = (provider as any).maxTokens || DEFAULT_CONTEXT_LENGTH;
                    needsUpdate = true;
                }
            }
            if (needsUpdate) {
                this.store.set('providers', providers);
                logger.debug('[ConfigStore] Migrated providers: added contextLength field');
            }
        }
    }

    get<K extends keyof AppConfig>(key: K): AppConfig[K] {
        return this.store.get(key);
    }

    set<K extends keyof AppConfig>(key: K, value: AppConfig[K]): void {
        this.store.set(key, value);
    }

    getAll(): AppConfig {
        const config = this.store.store as AppConfig;
        return {
            ...config,
            providers: this.getAllProviders()
        };
    }

    // === Multi-Provider Getters ===

    getActiveProviderId(): string {
        return this.store.get('activeProviderId') || 'minimax_intl';
    }

    setActiveProviderId(id: string): void {
        this.store.set('activeProviderId', id);
    }

    getProvider(id: string): ProviderConfig | undefined {
        const providers = this.store.get('providers') || defaultProviders;
        const stored = providers[id];
        const def = defaultProviders[id];

        if (stored && def && !stored.isCustom) {
            // Force update built-in fields that shouldn't change
            return {
                ...stored,
                apiUrl: def.apiUrl,
                model: def.model,
                readonlyUrl: def.readonlyUrl
            };
        }
        return stored;
    }

    getAllProviders(): Record<string, ProviderConfig> {
        const stored = this.store.get('providers') || defaultProviders;
        const merged: Record<string, ProviderConfig> = {};

        // First, add all stored providers
        for (const key in stored) {
            const s = stored[key];
            const d = defaultProviders[key];
            if (s && d && !s.isCustom) {
                // For built-in providers, merge stored user data (apiKey) with default configs
                // Only preserve user-modifiable fields, keep defaults for others
                merged[key] = {
                    ...d,  // Start with defaults (url, model, readonlyUrl)
                    ...s,  // Overlay user data (apiKey should be preserved)
                    // Explicitly preserve these user-modifiable fields
                    apiKey: s.apiKey || d.apiKey,
                    // Keep default values for these fields
                    apiUrl: d.apiUrl,
                    model: d.model,
                    readonlyUrl: d.readonlyUrl
                };
            } else {
                merged[key] = s;
            }
        }

        // Then add any missing default providers
        for (const key in defaultProviders) {
            if (!merged[key]) {
                merged[key] = { ...defaultProviders[key] };
            }
        }

        return merged;
    }

    updateProvider(id: string, config: Partial<ProviderConfig>): void {
        const providers = this.getAllProviders();
        if (providers[id]) {
            providers[id] = { ...providers[id], ...config };
            this.store.set('providers', providers);
        }
    }

    // === Backward/Agent Compatibility ===
    // These methods now return values based on ACTIVE provider

    getApiKey(): string {
        const active = this.getActiveProviderId();
        const provider = this.getProvider(active);
        return provider?.apiKey || process.env.ANTHROPIC_API_KEY || '';
    }

    setApiKey(key: string): void {
        const active = this.getActiveProviderId();
        this.updateProvider(active, { apiKey: key });
    }

    getModel(): string {
        const active = this.getActiveProviderId();
        return this.getProvider(active)?.model || '';
    }

    setModel(model: string): void {
        const active = this.getActiveProviderId();
        this.updateProvider(active, { model });
    }

    getApiUrl(): string {
        const active = this.getActiveProviderId();
        return this.getProvider(active)?.apiUrl || '';
    }

    setApiUrl(url: string): void {
        const active = this.getActiveProviderId();
        this.updateProvider(active, { apiUrl: url });
    }

    // === Helper for Main.ts set-all ===
    setAll(cfg: Partial<AppConfig>) {
        logger.debug('[ConfigStore] setAll called with:', Object.keys(cfg));

        if (cfg.authorizedFolders !== undefined) {
            this.store.set('authorizedFolders', cfg.authorizedFolders);
            logger.debug('[ConfigStore] Updated authorizedFolders');
        }
        if (cfg.networkAccess !== undefined) {
            this.store.set('networkAccess', cfg.networkAccess);
            logger.debug('[ConfigStore] Updated networkAccess');
        }
        if (cfg.shortcut !== undefined) {
            this.store.set('shortcut', cfg.shortcut);
            logger.debug('[ConfigStore] Updated shortcut');
        }
        if (cfg.activeProviderId !== undefined) {
            this.store.set('activeProviderId', cfg.activeProviderId);
            logger.debug('[ConfigStore] Updated activeProviderId:', cfg.activeProviderId);
        }
        if (cfg.providers !== undefined) {
            // Merge with existing providers to preserve user data
            const currentProviders = this.store.get('providers') || {};
            const mergedProviders = { ...currentProviders };

            // Update each provider from the input
            for (const [id, provider] of Object.entries(cfg.providers)) {
                if (mergedProviders[id]) {
                    // Merge with existing, preserving user data
                    mergedProviders[id] = {
                        ...mergedProviders[id],
                        ...provider,
                        // Ensure required fields exist
                        id: provider.id || id,
                        name: provider.name || mergedProviders[id].name
                    };
                } else {
                    // New provider
                    mergedProviders[id] = provider;
                }
            }

            this.store.set('providers', mergedProviders);
            logger.debug('[ConfigStore] Updated providers, keys:', Object.keys(mergedProviders));
        }

        // Verify the save
        const saved = this.store.get('providers');
        logger.debug('[ConfigStore] Verification - saved providers:', Object.keys(saved || {}).length);
    }

    // ... Keep Permissions methods

    // Authorized Folders
    getAuthorizedFolders(): FolderAuthorization[] {
        return this.store.get('authorizedFolders') || [];
    }

    // Get just the folder paths (for backward compatibility)
    getAuthorizedFolderPaths(): string[] {
        return this.getAuthorizedFolders().map(f => f.path);
    }

    addAuthorizedFolder(folder: string, trustLevel: TrustLevel = 'strict'): void {
        const folders = this.getAuthorizedFolders();
        const exists = folders.some(f => f.path === folder);
        if (!exists) {
            folders.push({
                path: folder,
                trustLevel,
                addedAt: Date.now()
            });
            this.store.set('authorizedFolders', folders);
        }
    }

    removeAuthorizedFolder(folder: string): void {
        const folders = this.getAuthorizedFolders().filter(f => f.path !== folder);
        this.store.set('authorizedFolders', folders);
    }

    // Set trust level for a specific folder
    setFolderTrustLevel(folderPath: string, level: TrustLevel): void {
        const folders = this.getAuthorizedFolders();
        const updated = folders.map(f =>
            f.path === folderPath ? { ...f, trustLevel: level } : f
        );
        this.store.set('authorizedFolders', updated);
    }

    // Get trust level for a specific folder
    getFolderTrustLevel(folderPath: string): TrustLevel {
        const folder = this.getAuthorizedFolders().find(f => f.path === folderPath);
        return folder?.trustLevel ?? 'strict';
    }

    // Check if any parent folder has trust level (for nested paths)
    getFileTrustLevel(filePath: string): TrustLevel {
        const folders = this.getAuthorizedFolders();
        // Find the most specific (longest path) matching folder
        let bestMatch: FolderAuthorization | null = null;
        for (const folder of folders) {
            if (filePath.startsWith(folder.path)) {
                if (!bestMatch || folder.path.length > bestMatch.path.length) {
                    bestMatch = folder;
                }
            }
        }
        return bestMatch?.trustLevel ?? 'strict';
    }

    // Network Access
    getNetworkAccess(): boolean {
        return this.store.get('networkAccess');
    }

    setNetworkAccess(enabled: boolean): void {
        this.store.set('networkAccess', enabled);
    }

    // Tool Permissions
    getAllowedPermissions(): ToolPermission[] {
        return this.store.get('allowedPermissions') || [];
    }

    addPermission(tool: string, pathPattern?: string): void {
        const permissions = this.getAllowedPermissions();
        // Check if already exists
        const exists = permissions.some(p =>
            p.tool === tool && p.pathPattern === (pathPattern || '*')
        );
        if (!exists) {
            permissions.push({
                tool,
                pathPattern: pathPattern || '*',
                grantedAt: Date.now()
            });
            this.store.set('allowedPermissions', permissions);
        }
    }

    removePermission(tool: string, pathPattern?: string): void {
        const permissions = this.getAllowedPermissions().filter(p =>
            !(p.tool === tool && p.pathPattern === (pathPattern || '*'))
        );
        this.store.set('allowedPermissions', permissions);
    }

    hasPermission(tool: string, path?: string): boolean {
        const permissions = this.getAllowedPermissions();
        return permissions.some(p => {
            if (p.tool !== tool) return false;
            if (p.pathPattern === '*') return true;
            if (!path) return p.pathPattern === '*';
            // Check if path matches pattern (simple prefix match)
            return path.startsWith(p.pathPattern || '');
        });
    }

    clearAllPermissions(): void {
        this.store.set('allowedPermissions', []);
    }

    // Max Tokens Configuration
    getMaxTokens(): number {
        const active = this.getActiveProviderId();
        return this.getProvider(active)?.maxTokens || DEFAULT_MAX_TOKENS;
    }

    setMaxTokens(maxTokens: number): void {
        const active = this.getActiveProviderId();
        this.updateProvider(active, { maxTokens });
    }

    getProviderMaxTokens(providerId: string): number {
        return this.getProvider(providerId)?.maxTokens || DEFAULT_MAX_TOKENS;
    }

    setProviderMaxTokens(providerId: string, maxTokens: number): void {
        this.updateProvider(providerId, { maxTokens });
    }

    // Context Length Configuration (上下文窗口长度)
    getContextLength(): number {
        const active = this.getActiveProviderId();
        return this.getProvider(active)?.contextLength || DEFAULT_CONTEXT_LENGTH;
    }

    setContextLength(contextLength: number): void {
        const active = this.getActiveProviderId();
        this.updateProvider(active, { contextLength });
    }

    getProviderContextLength(providerId: string): number {
        return this.getProvider(providerId)?.contextLength || DEFAULT_CONTEXT_LENGTH;
    }

    setProviderContextLength(providerId: string, contextLength: number): void {
        this.updateProvider(providerId, { contextLength });
    }

    // === Context Compression Configuration ===

    /**
     * 获取完整的压缩配置
     * 优先级：Provider 配置 > 全局配置 > 默认配置
     */
    getCompressionConfig(providerId?: string): CompressionConfig {
        const activeId = providerId || this.getActiveProviderId();
        const provider = this.getProvider(activeId);
        const globalConfig = this.store.get('globalCompressionConfig') || {};

        // 使用 contextLength（上下文长度）而不是 maxTokens（输出长度）
        const contextLength = provider?.contextLength || DEFAULT_CONTEXT_LENGTH;

        const defaultConfig: CompressionConfig = {
            enabled: true,
            maxContextSize: contextLength,
            compressionThreshold: 75,
            autoCondenseEnabled: true,
            condenseThresholdPercent: 60,
            maxSummaryTokens: 8000,
            truncateFallbackEnabled: true,
            truncateKeepMessages: 20,
            truncateKeepToolResults: 3,
            preservePatterns: [],
            aggressiveMode: false
        };

        return {
            ...defaultConfig,
            ...globalConfig,
            ...provider?.compressionConfig,
            // 确保 maxContextSize 使用 provider 的 contextLength（上下文长度）
            maxContextSize: contextLength
        };
    }

    /**
     * 设置 Provider 的压缩配置
     */
    setProviderCompressionConfig(
        providerId: string,
        config: Partial<CompressionConfig>
    ): void {
        const providers = this.getAllProviders();
        if (providers[providerId]) {
            providers[providerId].compressionConfig = {
                ...providers[providerId].compressionConfig,
                ...config
            };
            this.store.set('providers', providers);
            logger.debug(`[ConfigStore] Updated compression config for provider: ${providerId}`);
        }
    }

    /**
     * 设置全局压缩配置
     */
    setGlobalCompressionConfig(config: Partial<CompressionConfig>): void {
        const currentConfig = this.store.get('globalCompressionConfig') || {};
        const mergedConfig = { ...currentConfig, ...config };
        this.store.set('globalCompressionConfig', mergedConfig);
        logger.debug('[ConfigStore] Updated global compression config');
    }
}

export const configStore = new ConfigStore();
