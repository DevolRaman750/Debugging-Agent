// ══════════════════════════════════════════════════════════════
// Log Models
// ══════════════════════════════════════════════════════════════
// Maps to Python: src/models/log.py → LogEntry

export enum LogLevel {
  TRACE = "TRACE",
  DEBUG = "DEBUG",
  INFO = "INFO",
  WARNING = "WARNING",
  ERROR = "ERROR",
  CRITICAL = "CRITICAL",
}

export interface LogEntry {
  time: number;
  function_name: string;
  level: LogLevel;
  message: string;
  file_name: string;
  line_number: number;
  trace_id?: string;
  span_id?: string;
  git_url?: string;
  commit_id?: string;
  line?: string;                // Source code line content
  lines_above?: string[];       // Context lines above
  lines_below?: string[];       // Context lines below
}

/** SpanLog: { [spanId]: LogEntry[] } */
export interface SpanLog {
  [spanId: string]: LogEntry[];
}

/** TraceLog: { [traceId]: SpanLog[] } */
export interface TraceLog {
  [traceId: string]: SpanLog[];
}

export interface LogResponse {
  success: boolean;
  logs: { logs: LogEntry[] };
  error?: string;
}

/** Log level → color mapping for UI rendering */
export const LOG_LEVEL_COLORS: Record<LogLevel, string> = {
  [LogLevel.CRITICAL]: "#7f1d1d",
  [LogLevel.ERROR]: "#dc2626",
  [LogLevel.WARNING]: "#fb923c",
  [LogLevel.INFO]: "#64748b",
  [LogLevel.DEBUG]: "#a855f7",
  [LogLevel.TRACE]: "#6366f1",
};
