"use client";

import React, { useMemo } from "react";
import { Span } from "@/models/trace";

interface WaterfallRow {
  span: Span;
  depth: number;
}

interface TraceDepthMiniMapProps {
  rows: WaterfallRow[];
  timelineStart: number;
  timelineDuration: number;
}

function hashString(input: string): number {
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash << 5) - hash + input.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function serviceFromSpan(span: Span): string {
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

function colorForService(service: string): string {
  const hue = hashString(service) % 360;
  return `hsl(${hue}, 68%, 58%)`;
}

export default function TraceDepthMiniMap({
  rows,
  timelineStart,
  timelineDuration,
}: TraceDepthMiniMapProps) {
  const maxDepth = useMemo(() => {
    if (rows.length === 0) {
      return 0;
    }
    return rows.reduce((max, row) => (row.depth > max ? row.depth : max), 0);
  }, [rows]);

  if (rows.length === 0) {
    return null;
  }

  return (
    <div className="bg-white dark:bg-zinc-900 rounded-lg border border-zinc-200 dark:border-zinc-700 p-2">
      <div className="text-[11px] font-medium text-zinc-600 dark:text-zinc-300 mb-2">
        Trace Depth Minimap
      </div>
      <div className="space-y-1">
        {Array.from({ length: maxDepth + 1 }).map((_, lane) => (
          <div
            key={`lane-${lane}`}
            className="relative h-2 rounded bg-zinc-100 dark:bg-zinc-800"
          >
            {rows
              .filter((row) => row.depth === lane)
              .map((row) => {
                const left =
                  ((row.span.start_time - timelineStart) / timelineDuration) * 100;
                const width = (row.span.duration / timelineDuration) * 100;
                const normalizedLeft = Math.max(0, Math.min(100, left));
                const normalizedWidth = Math.max(
                  0.35,
                  Math.min(100 - normalizedLeft, width),
                );

                return (
                  <div
                    key={`mini-${row.span.id}`}
                    className="absolute top-0 h-full rounded"
                    style={{
                      left: `${normalizedLeft}%`,
                      width: `${normalizedWidth}%`,
                      backgroundColor: colorForService(serviceFromSpan(row.span)),
                    }}
                    title={`${row.span.name}`}
                  />
                );
              })}
          </div>
        ))}
      </div>
    </div>
  );
}
