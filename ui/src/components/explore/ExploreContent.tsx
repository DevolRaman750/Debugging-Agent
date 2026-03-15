"use client";

import React from "react";
import ExploreHeader from "./ExploreHeader";
import { Trace } from "./Trace";
import RightPanelSwitch from "@/components/right-panel/RightPanelSwitch";
import AgentPanel from "@/components/agent-panel/AgentPanel";
import ResizablePanelComponent from "@/components/resizable/ResizablePanel";
import { useExplore } from "@/hooks/useExploreContext";

export default function ExploreContent() {
  const {
    selectedTraceIds,
    selectedSpanIds,
    traceQueryStartTime,
    traceQueryEndTime,
    allTraces,
    logSearchValue,
    metadataSearchTerms,
    selectTraces,
    selectSpans,
    viewType,
    setViewType,
  } = useExplore();

  const leftPanel = (
    <div className="flex flex-col h-full overflow-hidden">
      <Trace />
    </div>
  );

  const rightPanel = (
    <div className="flex flex-col h-full overflow-hidden">
      <RightPanelSwitch
        viewType={viewType}
        traceIds={selectedTraceIds}
        spanIds={selectedSpanIds}
        allTraces={allTraces}
        traceQueryStartTime={traceQueryStartTime}
        traceQueryEndTime={traceQueryEndTime}
        logSearchValue={logSearchValue}
        metadataSearchTerms={metadataSearchTerms}
        onTraceSelect={selectTraces}
        onSpanSelect={selectSpans}
        onViewTypeChange={setViewType}
      />
    </div>
  );

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <ExploreHeader />

      <div className="flex-1 overflow-hidden">
        <AgentPanel>
          <ResizablePanelComponent
            leftPanel={leftPanel}
            rightPanel={rightPanel}
            defaultLeftWidth={45}
            minLeftWidth={30}
            maxLeftWidth={60}
          />
        </AgentPanel>
      </div>
    </div>
  );
}
