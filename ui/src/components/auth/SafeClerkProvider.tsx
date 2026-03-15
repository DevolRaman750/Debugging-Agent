"use client";

import React from "react";
import { ClerkProvider } from "@clerk/nextjs";
import { dark } from "@clerk/themes";

/**
 * Wraps children with ClerkProvider only when a valid publishable key is
 * available at build/runtime. When no key is set (local dev), renders
 * children directly — individual hooks must handle the missing-provider
 * case via the `useSafeAuth` hook from `@/hooks/useSafeAuth`.
 */
export default function SafeClerkProvider({
  children,
  publishableKey,
}: {
  children: React.ReactNode;
  publishableKey?: string;
}) {
  // Clerk validates the key format; only pass to ClerkProvider if present
  if (!publishableKey) {
    return <>{children}</>;
  }

  return (
    <ClerkProvider
      publishableKey={publishableKey}
      appearance={{
        baseTheme: dark,
        variables: {
          colorPrimary: "hsl(220, 70%, 51%)",
          colorBackground: "hsl(224, 71%, 4%)",
          colorText: "hsl(210, 20%, 98%)",
          colorInputBackground: "hsl(215, 28%, 17%)",
          colorInputText: "hsl(210, 20%, 98%)",
        },
      }}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      afterSignOutUrl="/sign-in"
    >
      {children}
    </ClerkProvider>
  );
}
