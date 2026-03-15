// ══════════════════════════════════════════════════════════════
// Provider Utilities
// ══════════════════════════════════════════════════════════════

const STORAGE_KEY = "traceroot_provider_config";
const SELECTION_PREFIX = "traceroot_provider_selection";
const SPECIFIC_CONFIG_PREFIX = "traceroot_provider_specific";

const LOCAL_MODE = process.env.NEXT_PUBLIC_LOCAL_MODE === "true";

export type ProviderType = "trace" | "log";
export type ProviderName = "aws" | "tencent" | "jaeger";

export interface ProviderConfig {
  trace_provider: string;
  log_provider: string;
  trace_region?: string;
  log_region?: string;
}

interface URLProviderConfig {
  traceProvider?: string;
  traceRegion?: string;
  logProvider?: string;
  logRegion?: string;
}

const DEFAULTS: ProviderConfig = {
  trace_provider: "jaeger",
  log_provider: "jaeger",
};

const TRACE_REGION_DEFAULTS: Record<string, string> = {
  aws: "us-west-2",
  tencent: "ap-hongkong",
  jaeger: "local",
};

const LOG_REGION_DEFAULTS: Record<string, string> = {
  aws: "us-west-2",
  tencent: "ap-hongkong",
  jaeger: "local",
};

const getSelectionKey = (providerType: ProviderType): string =>
  `${SELECTION_PREFIX}_${providerType}`;

const getSpecificConfigKey = (
  providerType: ProviderType,
  provider: string,
): string => `${SPECIFIC_CONFIG_PREFIX}_${providerType}_${provider}`;

const safeParse = <T>(raw: string | null, fallback: T): T => {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
};

export const getJaegerEndpoint = (): string =>
  process.env.NEXT_PUBLIC_JAEGER_ENDPOINT || "http://localhost:16686";

export const getUserEmail = (): string | null => {
  if (typeof window === "undefined") return null;
  const userRaw = localStorage.getItem("user");
  if (!userRaw) return null;
  try {
    const parsed = JSON.parse(userRaw) as { email?: string };
    return parsed.email || null;
  } catch {
    return null;
  }
};

export const copyToClipboard = async (value: string): Promise<boolean> => {
  if (!value || typeof navigator === "undefined" || !navigator.clipboard) {
    return false;
  }
  try {
    await navigator.clipboard.writeText(value);
    return true;
  } catch {
    return false;
  }
};

export function loadProviderConfig(): ProviderConfig {
  if (typeof window === "undefined") return DEFAULTS;
  const config = safeParse<ProviderConfig>(
    localStorage.getItem(STORAGE_KEY),
    DEFAULTS,
  );
  return { ...DEFAULTS, ...config };
}

export function saveProviderConfig(config: ProviderConfig): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...DEFAULTS, ...config }));
}

export function readProvidersFromURL(): URLProviderConfig {
  if (typeof window === "undefined") return {};
  const params = new URLSearchParams(window.location.search);
  return {
    traceProvider:
      params.get("trace_provider") || params.get("traceProvider") || undefined,
    traceRegion:
      params.get("trace_region") || params.get("traceRegion") || undefined,
    logProvider:
      params.get("log_provider") || params.get("logProvider") || undefined,
    logRegion: params.get("log_region") || params.get("logRegion") || undefined,
  };
}

export function writeProvidersToURL(config: URLProviderConfig): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);

  if (config.traceProvider) url.searchParams.set("trace_provider", config.traceProvider);
  if (config.traceRegion) url.searchParams.set("trace_region", config.traceRegion);
  if (config.logProvider) url.searchParams.set("log_provider", config.logProvider);
  if (config.logRegion) url.searchParams.set("log_region", config.logRegion);

  window.history.replaceState({}, "", url.toString());
}

export function saveProviderSelection(
  providerType: ProviderType,
  provider: string,
): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(getSelectionKey(providerType), provider);
}

export function clearProviderSelection(providerType: ProviderType): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(getSelectionKey(providerType));
}

export function loadProviderSelection(
  providerType?: ProviderType,
): string | ProviderConfig | null {
  if (typeof window === "undefined") {
    return providerType ? null : DEFAULTS;
  }
  if (!providerType) return loadProviderConfig();
  return localStorage.getItem(getSelectionKey(providerType));
}

export function getProviderRegion(
  providerType: ProviderType,
  provider: string,
): string {
  const loaded = loadProviderConfig();

  if (providerType === "trace" && loaded.trace_provider === provider && loaded.trace_region) {
    return loaded.trace_region;
  }
  if (providerType === "log" && loaded.log_provider === provider && loaded.log_region) {
    return loaded.log_region;
  }

  return providerType === "trace"
    ? TRACE_REGION_DEFAULTS[provider] || TRACE_REGION_DEFAULTS.jaeger
    : LOG_REGION_DEFAULTS[provider] || LOG_REGION_DEFAULTS.jaeger;
}

export function buildProviderParams(): URLSearchParams {
  const config = loadProviderConfig();
  const params = new URLSearchParams();

  params.set("trace_provider", config.trace_provider || DEFAULTS.trace_provider);
  params.set("log_provider", config.log_provider || DEFAULTS.log_provider);
  if (config.trace_region) params.set("trace_region", config.trace_region);
  if (config.log_region) params.set("log_region", config.log_region);

  return params;
}

export function appendProviderParams(url: string): string {
  const params = buildProviderParams();
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}${params.toString()}`;
}

export async function saveSpecificProviderConfig(
  providerType: ProviderType,
  provider: string,
  config: Record<string, unknown>,
): Promise<void> {
  if (typeof window === "undefined") return;
  localStorage.setItem(
    getSpecificConfigKey(providerType, provider),
    JSON.stringify(config),
  );
}

export function deleteSpecificProviderConfig(
  providerType: ProviderType,
  provider: string,
): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(getSpecificConfigKey(providerType, provider));
}

export async function loadAllProviderConfigs(
  providerType: ProviderType,
): Promise<Record<string, any>> {
  if (typeof window === "undefined") return {};

  const aws = safeParse<Record<string, any>>(
    localStorage.getItem(getSpecificConfigKey(providerType, "aws")),
    {},
  );
  const tencent = safeParse<Record<string, any>>(
    localStorage.getItem(getSpecificConfigKey(providerType, "tencent")),
    {},
  );
  const jaeger = safeParse<Record<string, any>>(
    localStorage.getItem(getSpecificConfigKey(providerType, "jaeger")),
    {},
  );

  const selected = loadProviderSelection(providerType);

  if (providerType === "trace") {
    return {
      traceProvider: selected,
      ...aws,
      ...tencent,
      ...jaeger,
    };
  }

  return {
    logProvider: selected,
    ...aws,
    ...tencent,
    ...jaeger,
  };
}

export function applyTraceConfig(
  config: any,
  setters: {
    setTencentRegion: (v: string) => void;
    setTencentSecretId: (v: string) => void;
    setTencentSecretKey: (v: string) => void;
    setJaegerEndpoint: (v: string) => void;
    setTencentInstanceId: (v: string) => void;
  },
): void {
  if (!config) return;

  if (config.tencentTraceConfig) {
    setters.setTencentRegion(config.tencentTraceConfig.region || "ap-hongkong");
    setters.setTencentSecretId(config.tencentTraceConfig.secretId || "");
    setters.setTencentSecretKey(config.tencentTraceConfig.secretKey || "");
    setters.setTencentInstanceId(config.tencentTraceConfig.apmInstanceId || "");
  }

  if (config.jaegerTraceConfig?.endpoint) {
    setters.setJaegerEndpoint(config.jaegerTraceConfig.endpoint);
  }
}

export function applyLogConfig(
  config: any,
  setters: {
    setTencentRegion: (v: string) => void;
    setTencentSecretId: (v: string) => void;
    setTencentSecretKey: (v: string) => void;
    setJaegerEndpoint: (v: string) => void;
    setTencentTopicId: (v: string) => void;
  },
): void {
  if (!config) return;

  if (config.tencentLogConfig) {
    setters.setTencentRegion(config.tencentLogConfig.region || "ap-hongkong");
    setters.setTencentSecretId(config.tencentLogConfig.secretId || "");
    setters.setTencentSecretKey(config.tencentLogConfig.secretKey || "");
    setters.setTencentTopicId(config.tencentLogConfig.clsTopicId || "");
  }

  if (config.jaegerLogConfig?.endpoint) {
    setters.setJaegerEndpoint(config.jaegerLogConfig.endpoint);
  }
}

export function initializeProviders(): {
  traceProvider: string;
  traceRegion: string;
  logProvider: string;
  logRegion: string;
} {
  if (LOCAL_MODE) {
    const local = {
      traceProvider: "jaeger",
      traceRegion: "local",
      logProvider: "jaeger",
      logRegion: "local",
    };

    saveProviderConfig({
      trace_provider: local.traceProvider,
      log_provider: local.logProvider,
      trace_region: local.traceRegion,
      log_region: local.logRegion,
    });

    return local;
  }

  const urlConfig = readProvidersFromURL();
  const savedConfig = loadProviderConfig();
  const localTraceProvider = loadProviderSelection("trace") as string | null;
  const localLogProvider = loadProviderSelection("log") as string | null;

  const traceProvider =
    urlConfig.traceProvider ||
    localTraceProvider ||
    savedConfig.trace_provider ||
    DEFAULTS.trace_provider;

  const logProvider =
    urlConfig.logProvider ||
    localLogProvider ||
    savedConfig.log_provider ||
    DEFAULTS.log_provider;

  const traceRegion =
    urlConfig.traceRegion ||
    savedConfig.trace_region ||
    getProviderRegion("trace", traceProvider);

  const logRegion =
    urlConfig.logRegion ||
    savedConfig.log_region ||
    getProviderRegion("log", logProvider);

  saveProviderSelection("trace", traceProvider);
  saveProviderSelection("log", logProvider);
  saveProviderConfig({
    trace_provider: traceProvider,
    log_provider: logProvider,
    trace_region: traceRegion,
    log_region: logRegion,
  });

  writeProvidersToURL({
    traceProvider,
    traceRegion,
    logProvider,
    logRegion,
  });

  return { traceProvider, traceRegion, logProvider, logRegion };
}
