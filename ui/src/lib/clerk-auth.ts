import { auth, currentUser } from "@clerk/nextjs/server";

// ══════════════════════════════════════════════════════════════
// Clerk Auth Helpers — Server-Side Only
// ══════════════════════════════════════════════════════════════
// Centralized auth utilities for API routes and server components.
// Gracefully degrades when Clerk keys are not configured (local dev).

/** Whether Clerk is configured with valid keys. */
export function isClerkConfigured(): boolean {
  return !!(
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY &&
    process.env.CLERK_SECRET_KEY
  );
}

/** Authenticated user info (or dev defaults when Clerk is not configured). */
export interface AuthUser {
  userId: string;
  email: string;
  fullName: string | null;
  imageUrl: string | null;
  isAuthenticated: boolean;
}

/** Dev-mode fallback user. */
const DEV_USER: AuthUser = {
  userId: "dev_user",
  email: "dev@traceroot.local",
  fullName: "Dev User",
  imageUrl: null,
  isAuthenticated: false,
};

/**
 * Get the current authenticated user.
 *
 * - With Clerk configured: returns real user data from session.
 * - Without Clerk: returns a deterministic dev user so the app runs locally.
 *
 * @throws Never — always returns a user (real or dev fallback).
 */
export async function getAuthUser(): Promise<AuthUser> {
  if (!isClerkConfigured()) {
    return DEV_USER;
  }

  try {
    const { userId } = await auth();

    if (!userId) {
      return DEV_USER;
    }

    const user = await currentUser();

    return {
      userId,
      email:
        user?.emailAddresses?.[0]?.emailAddress ?? "unknown@traceroot.local",
      fullName:
        [user?.firstName, user?.lastName].filter(Boolean).join(" ") || null,
      imageUrl: user?.imageUrl ?? null,
      isAuthenticated: true,
    };
  } catch {
    // Clerk SDK error (e.g., invalid keys) — fall back to dev user.
    return DEV_USER;
  }
}

/**
 * Get the Clerk session JWT token for backend API calls.
 *
 * @returns Bearer token string, or null if unavailable.
 */
export async function getSessionToken(): Promise<string | null> {
  if (!isClerkConfigured()) {
    return null;
  }

  try {
    const { getToken } = await auth();
    return await getToken();
  } catch {
    return null;
  }
}

/**
 * Require authentication — throws if user is not authenticated.
 * Use this in API routes that must be protected.
 *
 * @throws Error with 401-appropriate message.
 */
export async function requireAuth(): Promise<AuthUser> {
  if (!isClerkConfigured()) {
    return DEV_USER; // Allow in dev mode
  }

  const user = await getAuthUser();

  if (!user.isAuthenticated) {
    throw new Error("Authentication required");
  }

  return user;
}

// ── Request-level auth helpers (used by proxy API routes) ────

/** Result of validating auth on an incoming request. */
export interface AuthTokenAndHeaders {
  user: AuthUser;
  token: string | null;
}

/**
 * Validate the current request's auth and return user + token.
 *
 * Returns `null` when Clerk is configured but the user is not
 * authenticated (caller should return 401).
 * In dev mode (no Clerk keys) always returns the DEV_USER.
 */
export async function getAuthTokenAndHeaders(
  _request?: Request,
): Promise<AuthTokenAndHeaders | null> {
  const [user, token] = await Promise.all([
    getAuthUser(),
    getSessionToken(),
  ]);

  // If Clerk is active but user is not authenticated → deny.
  if (isClerkConfigured() && !user.isAuthenticated) {
    return null;
  }

  return { user, token };
}

/**
 * Build `HeadersInit` for a fetch() call to the Python backend.
 * Accepts the result of `getAuthTokenAndHeaders()`.
 */
export function createFetchHeaders(
  authResult: AuthTokenAndHeaders,
): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (authResult.token) {
    headers["Authorization"] = `Bearer ${authResult.token}`;
  }

  headers["X-User-Email"] = authResult.user.email;
  headers["X-User-Sub"] = authResult.user.userId;

  return headers;
}
