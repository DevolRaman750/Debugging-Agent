"use client";

import React, { useEffect, useState } from "react";
import LogPanelSwitch from "./log/LogPanelSwitch";
import TracePanelSwitch from "./trace/TracePanelSwitch";
import { ViewType } from "./ModeToggle";
import { Span, Trace as TraceModel } from "@/models/trace";

interface RightPanelSwitchProps {
  viewType: ViewType;
  traceIds: string[];
  spanIds: string[];
  allTraces: TraceModel[];
  traceQueryStartTime?: Date;
  traceQueryEndTime?: Date;
  logSearchValue?: string;
  metadataSearchTerms?: { category: string; value: string }[];
  onTraceSelect?: (traceIds: string[]) => void;
  onSpanSelect?: (spanIds: string[]) => void;
  onViewTypeChange?: (type: ViewType) => void;
}

export default function RightPanelSwitch({
  viewType,
  traceIds,
  spanIds,
  allTraces,
  traceQueryStartTime,
  traceQueryEndTime,
  logSearchValue = "",
  metadataSearchTerms = [],
  onTraceSelect,
  onSpanSelect,
  onViewTypeChange,
}: RightPanelSwitchProps) {
  const [spans, setSpans] = useState<Span[] | undefined>(undefined);
  const [spanTimeWindow, setSpanTimeWindow] = useState<
    { start: Date; end: Date } | undefined
  >(undefined);
  const [traceDurations, setTraceDurations] = useState<number[]>([]);
  const [traceStartTimes, setTraceStartTimes] = useState<Date[]>([]);
  const [traceEndTimes, setTraceEndTimes] = useState<Date[]>([]);
  const [traceIDs, setTraceIDs] = useState<string[]>([]);
  const [tracePercentiles, setTracePercentiles] = useState<string[]>([]);

  useEffect(() => {
    if (spanIds.length === 0) {
      setSpanTimeWindow(undefined);
    }
  }, [spanIds]);

  const handleSpanDrilldown = (span: Span) => {
    onSpanSelect?.([span.id]);
    setSpanTimeWindow({
      start: new Date(span.start_time * 1000),
      end: new Date(span.end_time * 1000),
    });
    onViewTypeChange?.("log");
  };

  // Job 2: Pre-compute arrays used by overview/detail child views.
  useEffect(() => {
    if (allTraces.length === 0) {
      setTraceDurations([]);
      setTraceStartTimes([]);
      setTraceEndTimes([]);
      setTraceIDs([]);
      setTracePercentiles([]);
      return;
    }

    setTraceDurations(allTraces.map((t) => t.duration));
    setTraceStartTimes(allTraces.map((t) => new Date(t.start_time * 1000)));
    setTraceEndTimes(allTraces.map((t) => new Date(t.end_time * 1000)));
    setTraceIDs(allTraces.map((t) => t.id));
    setTracePercentiles(allTraces.map((t) => t.percentile));
  }, [allTraces]);

  // Job 1: Resolve selected-trace spans. Merge spans for multi-trace selection.
  useEffect(() => {
    if (traceIds.length === 0 || allTraces.length === 0) {
      setSpans(undefined);
      return;
    }

    if (traceIds.length === 1) {
      const trace = allTraces.find((t) => t.id === traceIds[0]);
      setSpans(trace?.spans);
      return;
    }

    const mergedSpans: Span[] = [];
    traceIds.forEach((traceId) => {
      const trace = allTraces.find((t) => t.id === traceId);
      if (trace?.spans) {
        mergedSpans.push(...trace.spans);
      }
    });

    setSpans(mergedSpans.length > 0 ? mergedSpans : undefined);
  }, [traceIds, allTraces]);

  // Job 3: Route to log or trace panel using prepared data.
  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 overflow-y-auto overflow-x-hidden">
        {viewType === "log" && (
          <LogPanelSwitch
            traceIds={traceIds}
            spanIds={spanIds}
            traceQueryStartTime={spanTimeWindow?.start ?? traceQueryStartTime}
            traceQueryEndTime={spanTimeWindow?.end ?? traceQueryEndTime}
            segments={spans}
            allTraces={allTraces}
            traceDurations={traceDurations}
            traceStartTimes={traceStartTimes}
            traceEndTimes={traceEndTimes}
            traceIDs={traceIDs}
            tracePercentiles={tracePercentiles}
            logSearchValue={logSearchValue}
            metadataSearchTerms={metadataSearchTerms}
            onTraceSelect={onTraceSelect}
            onSpanSelect={handleSpanDrilldown}
            viewType={viewType}
          />
        )}

        {viewType === "trace" && (
          <TracePanelSwitch
            traceId={traceIds.length === 1 ? traceIds[0] : undefined}
            spanIds={spanIds}
            traceQueryStartTime={traceQueryStartTime}
            traceQueryEndTime={traceQueryEndTime}
            segments={spans}
            traceDurations={traceDurations}
            traceStartTimes={traceStartTimes}
            traceEndTimes={traceEndTimes}
            traceIDs={traceIDs}
            tracePercentiles={tracePercentiles}
            onTraceSelect={onTraceSelect}
            onSpanSelect={handleSpanDrilldown}
          />
        )}
      </div>
    </div>
  );
}
