"use client";

import React from "react";
import SearchBar from "./SearchBar";
import RefreshButton from "./RefreshButton";
import TimeButton from "./TimeButton";
import ModeToggle from "../right-panel/ModeToggle";
import { useExplore } from "@/hooks/useExploreContext";

export default function ExploreHeader() {
  const {
    loading,
    selectedTimeRange,
    timezone,
    viewType,
    agentOpen,
    search,
    clearSearch,
    setLogSearchValue,
    setMetadataSearchTerms,
    selectTimeRange,
    selectCustomTimeRange,
    refresh,
    setViewType,
    toggleAgent,
  } = useExplore();

  return (
    <div className="sticky top-0 z-10 bg-white dark:bg-zinc-950 pt-1 pl-6 pr-2 pb-1 border-b border-zinc-200 dark:border-zinc-700">
      <div className="flex flex-row justify-between items-center gap-2">
        <div className="flex-1 min-w-0">
          <SearchBar
            onSearch={search}
            onClear={clearSearch}
            onLogSearchValueChange={setLogSearchValue}
            onMetadataSearchTermsChange={setMetadataSearchTerms}
            disabled={loading}
          />
        </div>
        <div className="flex items-center space-x-2 flex-shrink-0 justify-end">
          <RefreshButton onRefresh={refresh} disabled={loading} />
          <TimeButton
            selectedTimeRange={selectedTimeRange}
            onTimeRangeSelect={selectTimeRange}
            onCustomTimeRangeSelect={selectCustomTimeRange}
            currentTimezone={timezone}
            disabled={loading}
          />
          <ModeToggle
            viewType={viewType}
            onViewTypeChange={setViewType}
            agentOpen={agentOpen}
            onAgentToggle={toggleAgent}
          />
        </div>
      </div>
    </div>
  );
}
