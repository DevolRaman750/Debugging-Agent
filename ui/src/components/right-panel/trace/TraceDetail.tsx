"use client";

import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Span } from "@/models/trace";
import TraceDepthChart from "./TraceDepthChart";

interface TraceDetailProps {
  traceId?: string;
  spanIds?: string[];
  spans?: Span[];
  traceStartTime?: number;
  traceEndTime?: number;
  traceDuration?: number;
  percentile?: string;
  onSpanSelect?: (span: Span) => void;
}

interface WaterfallRow {
  span: Span;
  depth: number;
}

function formatDuration(seconds: number): string {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(1)}ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(3)}s`;
  }
  return `${(seconds / 60).toFixed(2)}m`;
}

function flattenSpanTree(rootSpans: Span[]): WaterfallRow[] {
  const rows: WaterfallRow[] = [];

  const walk = (items: Span[], depth: number) => {
    const sorted = [...items].sort((a, b) => a.start_time - b.start_time);
    sorted.forEach((span) => {
      rows.push({ span, depth });
      if (span.spans && span.spans.length > 0) {
        walk(span.spans, depth + 1);
      }
    });
  };

  walk(rootSpans, 0);
  return rows;
}

function getServiceName(span: Span): string {
  const unsafeSpan = span as Span & {
    service_name?: string;
    serviceName?: string;
    service?: string;
    attributes?: Record<string, unknown>;
  };

  const attrService =
    typeof unsafeSpan.attributes?.service_name === "string"
      ? unsafeSpan.attributes.service_name
      : undefined;

  return (
    unsafeSpan.service_name ||
    unsafeSpan.serviceName ||
    unsafeSpan.service ||
    attrService ||
    span.name ||
    "unknown"
  );
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function colorForService(serviceName: string): { background: string; border: string } {
  const hue = hashString(serviceName) % 360;
  return {
    background: `hsl(${hue}, 74%, 82%)`,
    border: `hsl(${hue}, 62%, 40%)`,
  };
}

export default function TraceDetail({
  traceId,
  spanIds = [],
  spans = [],
  traceStartTime,
  traceEndTime,
  traceDuration,
  percentile,
  onSpanSelect,
}: TraceDetailProps) {
  const rows = useMemo(() => flattenSpanTree(spans), [spans]);

  const timeline = useMemo(() => {
    if (rows.length === 0) {
      return null;
    }

    let minStart = rows[0].span.start_time;
    let maxEnd = rows[0].span.end_time;

    rows.forEach(({ span }) => {
      const spanEnd = Math.max(span.end_time, span.start_time + span.duration);
      minStart = Math.min(minStart, span.start_time);
      maxEnd = Math.max(maxEnd, spanEnd);
    });

    if (typeof traceStartTime === "number") {
      minStart = Math.min(minStart, traceStartTime);
    }

    if (typeof traceEndTime === "number") {
      maxEnd = Math.max(maxEnd, traceEndTime);
    }

    if (typeof traceDuration === "number" && traceDuration > 0) {
      maxEnd = Math.max(maxEnd, minStart + traceDuration);
    }

    const duration = Math.max(maxEnd - minStart, 0.000001);

    return {
      start: minStart,
      end: maxEnd,
      duration,
    };
  }, [rows, traceStartTime, traceEndTime, traceDuration]);

  const selectedSpanIds = useMemo(() => new Set(spanIds), [spanIds]);

  if (!traceId || rows.length === 0 || !timeline) {
    return (
      <div className="h-full flex flex-col">
        <div className="bg-white dark:bg-zinc-900 p-3 overflow-y-auto flex-1 min-h-0">
          <div className="p-3 rounded-lg border border-zinc-200 dark:border-zinc-700">
            <p className="text-sm text-zinc-600 dark:text-zinc-300">
              {!traceId ? "No trace selected" : "No spans found for this trace"}
            </p>
          </div>
        </div>
      </div>
    );
  }

  const tickCount = 8;
  const ticks = Array.from({ length: tickCount + 1 }, (_, index) => {
    const ratio = index / tickCount;
    const t = timeline.duration * ratio;
    return {
      left: ratio * 100,
      label: formatDuration(t),
    };
  });

  return (
    <div className="h-full flex flex-col overflow-y-auto">
      <div className="bg-zinc-50 dark:bg-zinc-900 mt-1 ml-4 mr-4 mb-4 rounded-lg flex flex-col min-h-0 flex-1 p-2 gap-2">
        <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700 p-2 flex flex-wrap items-center gap-2">
          {percentile ? <Badge className="h-6">{percentile}</Badge> : null}
          <Badge variant="secondary" className="h-6">Trace: {traceId}</Badge>
          <Badge variant="outline" className="h-6">
            Duration: {formatDuration(timeline.duration)}
          </Badge>
          <Badge variant="secondary" className="h-6">
            Spans: {rows.length}
          </Badge>
        </div>

        <TraceDepthChart
          rows={rows}
          timelineStart={timeline.start}
          timelineDuration={timeline.duration}
        />

        <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700 min-h-0 flex-1 overflow-hidden">
          <div className="px-3 pt-3 pb-1 border-b border-zinc-200 dark:border-zinc-700">
            <div className="relative h-7">
              <div className="absolute top-0 left-0 right-0 h-px bg-zinc-300 dark:bg-zinc-700" />
              {ticks.map((tick, index) => {
                const alignClass =
                  index === 0
                    ? "translate-x-0"
                    : index === ticks.length - 1
                      ? "-translate-x-full"
                      : "-translate-x-1/2";

                return (
                  <div
                    key={`${tick.left}-${index}`}
                    className="absolute top-0"
                    style={{ left: `${tick.left}%` }}
                  >
                    <div className={`w-px h-2 bg-zinc-400 dark:bg-zinc-500 ${alignClass}`} />
                    <div className={`text-[10px] text-zinc-500 dark:text-zinc-400 mt-1 whitespace-nowrap ${alignClass}`}>
                      {tick.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="h-full overflow-y-auto p-2">
            <div className="space-y-1.5 pb-24">
              {rows.map(({ span, depth }) => {
                const left = ((span.start_time - timeline.start) / timeline.duration) * 100;
                const width = (span.duration / timeline.duration) * 100;
                const normalizedLeft = Math.max(0, Math.min(100, left));
                const normalizedWidth = Math.max(0.6, Math.min(100 - normalizedLeft, width));
                const isSelected = selectedSpanIds.has(span.id);
                const serviceName = getServiceName(span);
                const spanColor = colorForService(serviceName);

                return (
                  <div key={span.id} className="grid grid-cols-[280px_1fr] gap-2 items-center">
                    <button
                      type="button"
                      className={`text-left text-xs rounded px-2 py-1 truncate border transition-colors ${
                        isSelected
                          ? "bg-zinc-100 dark:bg-zinc-800 border-zinc-400 dark:border-zinc-500"
                          : "bg-zinc-50 dark:bg-zinc-900 border-zinc-200 dark:border-zinc-700"
                      }`}
                      style={{ paddingLeft: `${8 + depth * 14}px` }}
                      onClick={() => onSpanSelect?.(span)}
                      title={span.name}
                    >
                      {span.name}
                    </button>

                    <div className="relative h-8 rounded border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-900">
                      <button
                        type="button"
                        className={`absolute top-1 bottom-1 rounded border px-2 text-[11px] leading-none overflow-hidden text-left transition-all ${
                          isSelected ? "ring-2 ring-zinc-900/20 dark:ring-zinc-100/20" : ""
                        }`}
                        style={{
                          left: `${normalizedLeft}%`,
                          width: `${normalizedWidth}%`,
                          backgroundColor: spanColor.background,
                          borderColor: spanColor.border,
                        }}
                        onClick={() => onSpanSelect?.(span)}
                        title={`${serviceName} - ${span.name} (${formatDuration(span.duration)})`}
                      >
                        <span className="inline-flex items-center gap-2 truncate">
                          <span className="font-medium truncate">{serviceName}</span>
                          <span className="opacity-80 truncate">{formatDuration(span.duration)}</span>
                        </span>
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
