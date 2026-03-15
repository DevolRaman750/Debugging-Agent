"use client";

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
} from "react";
import { Trace as TraceModel } from "@/models/trace";
import { SearchCriterion } from "@/components/explore/SearchBar";
import { TimeRange, TIME_RANGES } from "@/components/explore/TimeButton";
import {
  CustomTimeRange,
  TimezoneMode,
} from "@/components/explore/CustomTimeRangeDialog";
import { ViewType } from "@/components/right-panel/ModeToggle";

// ── State shape ──

export interface ExploreState {
  // Loading
  loading: boolean;
  // Time range
  selectedTimeRange: TimeRange;
  timezone: TimezoneMode;
  // Search
  searchCriteria: SearchCriterion[];
  logSearchValue: string;
  metadataSearchTerms: { category: string; value: string }[];
  // Selections
  selectedTraceIds: string[];
  selectedSpanIds: string[];
  // Trace data
  traceQueryStartTime: Date | undefined;
  traceQueryEndTime: Date | undefined;
  allTraces: TraceModel[];
  // View
  viewType: ViewType;
  agentOpen: boolean;
}

// ── Actions exposed by the context ──

export interface ExploreActions {
  setLoading: (loading: boolean) => void;
  // Time range
  selectTimeRange: (range: TimeRange) => void;
  selectCustomTimeRange: (
    customRange: CustomTimeRange,
    tz: TimezoneMode,
  ) => void;
  // Search
  search: (criteria: SearchCriterion[]) => void;
  clearSearch: () => void;
  setLogSearchValue: (value: string) => void;
  setMetadataSearchTerms: (
    terms: { category: string; value: string }[],
  ) => void;
  // Selections
  selectTraces: (traceIds: string[]) => void;
  selectSpans: (spanIds: string[]) => void;
  clearSpans: () => void;
  selectAgentSpan: (spanId: string) => void;
  // Trace data
  setTraceQueryTimes: (startTime: Date, endTime: Date) => void;
  updateTraces: (traces: TraceModel[]) => void;
  // View
  setViewType: (type: ViewType) => void;
  toggleAgent: () => void;
  // Refresh
  refresh: () => void;
}

type ExploreContextValue = ExploreState & ExploreActions;

const ExploreContext = createContext<ExploreContextValue | null>(null);

// ── Provider ──

export function ExploreProvider({ children }: { children: React.ReactNode }) {
  // Loading state
  const [loading, setLoading] = useState(false);

  // Time range state — default to "Last 3 Hours"
  const [selectedTimeRange, setSelectedTimeRange] = useState<TimeRange>(
    TIME_RANGES.find((r) => r.label === "Last 3 Hours") || TIME_RANGES[4],
  );
  const [timezone, setTimezone] = useState<TimezoneMode>("utc");

  // Search state
  const [searchCriteria, setSearchCriteria] = useState<SearchCriterion[]>([]);
  const [logSearchValue, setLogSearchValueState] = useState("");
  const [metadataSearchTerms, setMetadataSearchTermsState] = useState<
    { category: string; value: string }[]
  >([]);

  // Selection state
  const [selectedTraceIds, setSelectedTraceIds] = useState<string[]>([]);
  const [selectedSpanIds, setSelectedSpanIds] = useState<string[]>([]);

  // Trace data state
  const [traceQueryStartTime, setTraceQueryStartTime] = useState<
    Date | undefined
  >();
  const [traceQueryEndTime, setTraceQueryEndTime] = useState<
    Date | undefined
  >();
  const [allTraces, setAllTraces] = useState<TraceModel[]>([]);

  // View mode state
  const [viewType, setViewTypeState] = useState<ViewType>("trace");
  const [agentOpen, setAgentOpen] = useState(false);

  // ── Memoised action callbacks ──

  const selectTimeRange = useCallback((range: TimeRange) => {
    setSelectedTimeRange(range);
    setLoading(true);
  }, []);

  const selectCustomTimeRange = useCallback(
    (customRange: CustomTimeRange, tz: TimezoneMode) => {
      setTimezone(tz);
      setSelectedTimeRange({
        label: "Custom",
        isCustom: true,
        customRange,
      });
      setLoading(true);
    },
    [],
  );

  const search = useCallback((criteria: SearchCriterion[]) => {
    setSearchCriteria(criteria);
    setLoading(true);
  }, []);

  const clearSearch = useCallback(() => {
    setSearchCriteria([]);
    setLoading(true);
  }, []);

  const selectTraces = useCallback((traceIds: string[]) => {
    setSelectedTraceIds(traceIds);
  }, []);

  const selectSpans = useCallback((spanIds: string[]) => {
    setSelectedSpanIds(spanIds);
  }, []);

  const clearSpans = useCallback(() => {
    setSelectedSpanIds([]);
  }, []);

  const selectAgentSpan = useCallback((spanId: string) => {
    setSelectedSpanIds([spanId]);
  }, []);

  const setTraceQueryTimes = useCallback(
    (startTime: Date, endTime: Date) => {
      setTraceQueryStartTime(startTime);
      setTraceQueryEndTime(endTime);
    },
    [],
  );

  const updateTraces = useCallback((traces: TraceModel[]) => {
    setAllTraces(traces);
  }, []);

  const setViewType = useCallback((type: ViewType) => {
    setViewTypeState(type);
  }, []);

  const toggleAgent = useCallback(() => {
    setAgentOpen((prev) => !prev);
  }, []);

  const refresh = useCallback(() => {
    setLoading(true);
  }, []);

  // ── Build context value (memoised to avoid unnecessary re-renders) ──

  const value = useMemo<ExploreContextValue>(
    () => ({
      // state
      loading,
      selectedTimeRange,
      timezone,
      searchCriteria,
      logSearchValue,
      metadataSearchTerms,
      selectedTraceIds,
      selectedSpanIds,
      traceQueryStartTime,
      traceQueryEndTime,
      allTraces,
      viewType,
      agentOpen,
      // actions
      setLoading,
      selectTimeRange,
      selectCustomTimeRange,
      search,
      clearSearch,
      setLogSearchValue: setLogSearchValueState,
      setMetadataSearchTerms: setMetadataSearchTermsState,
      selectTraces,
      selectSpans,
      clearSpans,
      selectAgentSpan,
      setTraceQueryTimes,
      updateTraces,
      setViewType,
      toggleAgent,
      refresh,
    }),
    [
      loading,
      selectedTimeRange,
      timezone,
      searchCriteria,
      logSearchValue,
      metadataSearchTerms,
      selectedTraceIds,
      selectedSpanIds,
      traceQueryStartTime,
      traceQueryEndTime,
      allTraces,
      viewType,
      agentOpen,
      selectTimeRange,
      selectCustomTimeRange,
      search,
      clearSearch,
      selectTraces,
      selectSpans,
      clearSpans,
      selectAgentSpan,
      setTraceQueryTimes,
      updateTraces,
      setViewType,
      toggleAgent,
      refresh,
    ],
  );

  return (
    <ExploreContext.Provider value={value}>{children}</ExploreContext.Provider>
  );
}

// ── Consumer hook ──

export function useExplore(): ExploreContextValue {
  const ctx = useContext(ExploreContext);
  if (!ctx) {
    throw new Error("useExplore must be used within an <ExploreProvider>");
  }
  return ctx;
}
