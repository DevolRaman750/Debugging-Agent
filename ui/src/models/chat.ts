export type MessageType = "assistant" | "user" | "github" | "statistics";
export type ActionType = "agent_chat" | "pending_confirmation" | "github_create_issue" | "github_create_pr";
export type ActionStatus = "pending" | "success" | "failed" | "cancelled" | "awaiting_confirmation";

export type ChatMode = "agent" | "chat";
export type ChatModel = "gpt-4o" | "gpt-4.1" | "gpt-5" | "auto" | "llama-3.3-70b-versatile";
export type Provider = "openai" | "groq" | "custom";

export interface ChatProviders {
  trace_provider?: string;
  log_provider?: string;
  trace_region?: string;
  log_region?: string;
  provider?: Provider;
}

export interface Reference {
  number: number;
  span_id?: string;
  span_function_name?: string;
  line_number?: number;
  log_message?: string;
}

// NEW: Intelligence metadata from Stage 8
export interface IntelligenceMetadata {
  confidence: string;           // "HIGH" | "MEDIUM" | "LOW"
  pattern_matched: string | null;
  fast_path: boolean;
  processing_time_ms: number;
  top_cause: string | null;
  top_cause_score: number | null;
  causes_found: number;
}

export interface ChatbotResponse {
  time: number;
  message: string;
  reference: Reference[];
  message_type: MessageType;
  chat_id: string;
  action_type?: ActionType;
  status?: ActionStatus;
  metadata?: IntelligenceMetadata;  // ← NEW: from your response_builder
  user_feedback?: "positive" | "negative" | null;  // ← NEW: for feedback loop
}
export interface ChatRequest {
  time: number;
  message: string;
  message_type: MessageType;
  trace_id: string;
  span_ids: string[];
  start_time: number;
  end_time: number;
  model: ChatModel;
  mode: ChatMode;
  chat_id: string;
  trace_provider: string;
  log_provider: string;
  trace_region?: string;
  log_region?: string;
  provider: Provider;
  providers?: ChatProviders;
}

export interface ChatResponse {
  success: boolean;
  data: ChatbotResponse | null;
  error?: string;
}

export interface ChatHistoryResponse {
  history: ChatbotResponse[];
}
