import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Merge Tailwind classes with proper precedence.
 * Used by all shadcn/ui components.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format duration for display in trace list.
 * Converts seconds to human-readable (µs / ms / s / m).
 */
export function formatDuration(seconds: number): string {
  if (seconds < 0.001) {
    return `${(seconds * 1_000_000).toFixed(0)}µs`;
  }
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)}ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(2)}s`;
  }
  return `${(seconds / 60).toFixed(1)}m`;
}

/**
 * Truncate string with ellipsis.
 */
export function truncate(str: string, maxLength: number): string {
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength) + "...";
}

export function truncateTitle(str: string, maxLength: number = 40): string {
  return truncate(str, maxLength);
}
