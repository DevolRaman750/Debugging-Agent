// ══════════════════════════════════════════════════════════════
// Percentile Color Constants
// ══════════════════════════════════════════════════════════════

export type PercentileKey = "P50" | "P75" | "P90" | "P95" | "P99";

/** Percentile tag → Tailwind color class mapping */
export const PERCENTILE_COLORS: Record<string, string> = {
  P50: "bg-green-500/20 text-green-400 border-green-500/30",
  P75: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30",
  P90: "bg-orange-500/20 text-orange-400 border-orange-500/30",
  P95: "bg-red-500/20 text-red-400 border-red-500/30",
  P99: "bg-red-600/20 text-red-300 border-red-600/30",
};

export function getPercentileColor(percentile: PercentileKey): string {
  return PERCENTILE_COLORS[percentile] || PERCENTILE_COLORS.P50;
}

/** SDK language → icon/label mapping */
export const SDK_LANGUAGE_MAP: Record<string, { label: string; emoji: string }> = {
  python: { label: "Python", emoji: "🐍" },
  typescript: { label: "TypeScript", emoji: "TS" },
  javascript: { label: "JavaScript", emoji: "JS" },
  java: { label: "Java", emoji: "☕" },
  go: { label: "Go", emoji: "🔵" },
  rust: { label: "Rust", emoji: "🦀" },
  csharp: { label: "C#", emoji: "C#" },
};
