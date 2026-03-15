// ══════════════════════════════════════════════════════════════
// UUID Generator
// ══════════════════════════════════════════════════════════════

import { v4 as uuidv4 } from "uuid";

/**
 * Generate a hex UUID without dashes.
 * Used for chat_id generation matching Python backend format.
 */
export function generateUuidHex(): string {
  return uuidv4().replace(/-/g, "");
}
