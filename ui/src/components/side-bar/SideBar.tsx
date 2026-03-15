"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { useState, useCallback } from "react";
import {
  Compass,
  Plug,
  Settings,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";

// ── Types ────────────────────────────────────────────────────
interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/explore", label: "Explore", icon: Compass },
  { href: "/integrate", label: "Integrate", icon: Plug },
  { href: "/settings", label: "Settings", icon: Settings },
];

// ── Clerk UserButton (loaded conditionally) ──────────────────
function ClerkUserSlot() {
  // Only render <UserButton> when Clerk keys are configured.
  // At build-time / in dev without keys, the publishable key env var is empty,
  // so we show a static placeholder to avoid the ClerkProvider error.
  const clerkConfigured = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

  if (!clerkConfigured) {
    return (
      <div
        className="w-8 h-8 rounded-full bg-sidebar-accent flex items-center justify-center
                   text-xs font-medium text-sidebar-foreground select-none"
        title="Dev User"
      >
        DU
      </div>
    );
  }

  // Dynamic import so the Clerk bundle is only pulled in when configured.
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { UserButton } = require("@clerk/nextjs");
  return (
    <UserButton
      afterSignOutUrl="/sign-in"
      appearance={{
        elements: {
          avatarBox: "w-8 h-8",
        },
      }}
    />
  );
}

// ── SideBar Component ────────────────────────────────────────
export default function SideBar() {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);

  const toggleCollapse = useCallback(() => setCollapsed((c) => !c), []);

  /** Check if a nav item is the active route. */
  const isActive = (href: string) =>
    pathname === href || pathname.startsWith(`${href}/`);

  return (
    <aside
      className={`
        relative flex flex-col border-r border-sidebar-border bg-sidebar
        transition-[width] duration-200 ease-in-out
        ${collapsed ? "w-14" : "w-48"}
        h-screen shrink-0
      `}
    >
      {/* ── Logo ───────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-3 py-4">
        <Link
          href="/explore"
          className="flex items-center gap-2 min-w-0"
          aria-label="TraceRoot Home"
        >
          <div
            className="w-8 h-8 shrink-0 rounded-lg bg-primary/20
                        flex items-center justify-center
                        text-primary font-bold text-xs select-none"
          >
            TR
          </div>
          {!collapsed && (
            <span className="text-sm font-semibold text-sidebar-foreground truncate">
              TraceRoot
            </span>
          )}
        </Link>
      </div>

      {/* ── Navigation Links ───────────────────────────────── */}
      <nav className="flex-1 flex flex-col gap-1 px-2 mt-2" role="navigation">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = isActive(href);
          return (
            <Link
              key={href}
              href={href}
              title={collapsed ? label : undefined}
              className={`
                group flex items-center gap-3 rounded-md px-2.5 py-2
                text-sm font-medium transition-colors duration-150
                ${
                  active
                    ? "bg-sidebar-accent text-primary"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground"
                }
              `}
            >
              <Icon
                className={`w-4 h-4 shrink-0 ${
                  active ? "text-primary" : "text-sidebar-foreground/50 group-hover:text-sidebar-foreground"
                }`}
              />
              {!collapsed && <span className="truncate">{label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* ── Bottom Section: Collapse toggle + User ─────────── */}
      <div className="flex flex-col items-center gap-3 px-2 pb-4 mt-auto">
        {/* Collapse / Expand button */}
        <button
          onClick={toggleCollapse}
          className="w-full flex items-center justify-center rounded-md p-2
                     text-sidebar-foreground/50 hover:bg-sidebar-accent hover:text-sidebar-foreground
                     transition-colors duration-150"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? (
            <PanelLeft className="w-4 h-4" />
          ) : (
            <PanelLeftClose className="w-4 h-4" />
          )}
        </button>

        {/* Clerk User Button */}
        <ClerkUserSlot />
      </div>
    </aside>
  );
}
