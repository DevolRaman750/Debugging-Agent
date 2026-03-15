import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

// ── Dev-mode detection ───────────────────────────────────────
// clerkMiddleware() throws immediately if the publishable key is missing,
// so we must guard *before* even importing it.
const hasClerkKeys =
  !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY &&
  !!process.env.CLERK_SECRET_KEY;

// ── Build the middleware function ────────────────────────────
async function buildClerkMiddleware() {
  const { clerkMiddleware, createRouteMatcher } = await import(
    "@clerk/nextjs/server"
  );

  const isPublicRoute = createRouteMatcher([
    "/sign-in(.*)",
    "/sign-up(.*)",
    "/api/health(.*)",
  ]);

  return clerkMiddleware(async (auth, request) => {
    if (!isPublicRoute(request)) {
      await auth.protect();
    }
  });
}

// Cache the promise so we only build once.
const clerkMiddlewarePromise = hasClerkKeys ? buildClerkMiddleware() : null;

export default async function middleware(request: NextRequest) {
  // No Clerk keys → pass through (local dev mode).
  if (!clerkMiddlewarePromise) {
    return NextResponse.next();
  }

  const handler = await clerkMiddlewarePromise;
  return handler(request, {} as never);
}

export const config = {
  // Match all routes EXCEPT static files, _next internals, and common assets.
  matcher: [
    "/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)",
    "/(api|trpc)(.*)",
  ],
};
