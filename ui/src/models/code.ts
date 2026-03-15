// ══════════════════════════════════════════════════════════════
// Code Context Models
// ══════════════════════════════════════════════════════════════

export interface CodeResponse {
  line: string | null;            // The source code line at the error
  lines_above: string[] | null;   // Context lines above
  lines_below: string[] | null;   // Context lines below
  error_message: string | null;
}
