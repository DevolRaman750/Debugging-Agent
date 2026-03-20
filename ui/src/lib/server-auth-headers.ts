import { getAuthUser, getSessionToken } from "./clerk-auth";

// ══════════════════════════════════════════════════════════════
// Backend Auth Header Builder
// ══════════════════════════════════════════════════════════════
// Creates headers for Next.js API routes → Python backend calls.
// Uses centralized clerk-auth helpers. Gracefully degrades in dev.

/**
 * Build auth headers for requests to the Python FastAPI backend.
 *
 * In production (Clerk configured):
 *   Authorization: Bearer <clerk_jwt>
 *   X-User-Email: user@example.com
 *   X-User-Sub: user_clerk_id
 *
 * In development (no Clerk):
 *   X-User-Email: dev@rootix.local
 *   X-User-Sub: dev_user
 */
export async function createBackendAuthHeaders(
  _request?: Request,
): Promise<HeadersInit> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  const [user, token] = await Promise.all([
    getAuthUser(),
    getSessionToken(),
  ]);

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  headers["X-User-Email"] = user.email;
  headers["X-User-Sub"] = user.userId;

  return headers;
}
