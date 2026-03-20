// ══════════════════════════════════════════════════════════════
// LLM Model & Provider Constants
// ══════════════════════════════════════════════════════════════

import type { ChatModel, Provider } from "@/models/chat";
export type { ChatModel, Provider } from "@/models/chat";

export interface ModelOption {
  label: string;
  value: ChatModel;
  provider: Provider;
  description: string;
}

/**
 * Available LLM models.
 * The first model is the default.
 * Maps to Python backend: src/config.py → GROQ_MODEL
 */
export const AVAILABLE_MODELS: ModelOption[] = [
  {
    label: "Llama 3.3 70B",
    value: "llama-3.3-70b-versatile",
    provider: "groq",
    description: "Fast & capable (via Groq)",
  },
  {
    label: "GPT-4o",
    value: "gpt-4o",
    provider: "openai",
    description: "OpenAI flagship model",
  },
  {
    label: "Auto",
    value: "auto",
    provider: "openai",
    description: "Let Rootix choose the best model",
  },
];

export const DEFAULT_MODEL: ChatModel = AVAILABLE_MODELS[0].value;

export const DEFAULT_PROVIDER: Provider = "groq";

/**
 * Canonical frontend model map derived from AVAILABLE_MODELS.
 * Usage: CHAT_MODELS["gpt-4o"], CHAT_MODELS["llama-3.3-70b-versatile"], etc.
 */
export const CHAT_MODELS: Readonly<Record<ChatModel, ChatModel>> =
  AVAILABLE_MODELS.reduce(
    (acc, model) => {
      acc[model.value] = model.value;
      return acc;
    },
    {} as Record<ChatModel, ChatModel>,
  );

export function getModelsByProvider(provider: Provider): ChatModel[] {
  return AVAILABLE_MODELS.filter((m) => m.provider === provider).map(
    (m) => m.value,
  );
}

export const CHAT_MODEL_DISPLAY_NAMES: Record<ChatModel, string> = {
  "llama-3.3-70b-versatile": "Llama 3.3 70B",
  "gpt-4o": "GPT-4o",
  "gpt-4.1": "GPT-4.1",
  "gpt-5": "GPT-5",
  auto: "Auto",
};

/** Trace providers supported by the backend */
export const TRACE_PROVIDERS = ["jaeger", "aws", "tencent"] as const;
export type TraceProvider = (typeof TRACE_PROVIDERS)[number];

/** Log providers supported by the backend */
export const LOG_PROVIDERS = ["jaeger", "aws", "tencent"] as const;
export type LogProvider = (typeof LOG_PROVIDERS)[number];
