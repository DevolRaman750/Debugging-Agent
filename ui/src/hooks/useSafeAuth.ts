"use client";

import { useAuth as useClerkAuth, useUser as useClerkUser } from "@clerk/nextjs";

/**
 * Safe wrapper around Clerk's `useAuth`.
 * Returns sensible defaults when ClerkProvider is not mounted
 * (i.e. in local dev without Clerk keys).
 */
export function useSafeAuth() {
  try {
    return useClerkAuth();
  } catch {
    // ClerkProvider not mounted — return unauthenticated defaults
    return {
      isLoaded: true,
      isSignedIn: false,
      userId: null,
      sessionId: null,
      orgId: null,
      getToken: async () => null,
      signOut: async () => {},
    } as ReturnType<typeof useClerkAuth>;
  }
}

/**
 * Safe wrapper around Clerk's `useUser`.
 * Returns sensible defaults when ClerkProvider is not mounted.
 */
export function useSafeUser() {
  try {
    return useClerkUser();
  } catch {
    return {
      isLoaded: true,
      isSignedIn: false,
      user: null,
    } as ReturnType<typeof useClerkUser>;
  }
}
