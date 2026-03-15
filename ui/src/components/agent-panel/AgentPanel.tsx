"use client";

import React, { useState, useRef, useEffect } from "react";
import Agent from "@/components/right-panel/agent/Agent";
import { useExplore } from "@/hooks/useExploreContext";

interface AgentPanelProps {
  children: React.ReactNode;
}

export default function AgentPanel({ children }: AgentPanelProps) {
  const {
    selectedTraceIds,
    selectedSpanIds,
    traceQueryStartTime,
    traceQueryEndTime,
    agentOpen: isOpen,
    selectAgentSpan,
    setViewType,
    toggleAgent: onToggle,
  } = useExplore();
  const [panelWidth, setPanelWidth] = useState(30); // Track panel width percentage (default 30%)
  const [hasBeenOpened, setHasBeenOpened] = useState(false); // Track if panel has ever been opened
  const panelRef = useRef<HTMLDivElement>(null);

  const MIN_WIDTH = 20;
  const MAX_WIDTH = 35;

  // Track when panel has been opened at least once
  useEffect(() => {
    if (isOpen && !hasBeenOpened) {
      setHasBeenOpened(true);
    }
  }, [isOpen, hasBeenOpened]);

  // Handle ESC key to close panel
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onToggle();
      }
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [isOpen, onToggle]);

  // Handle Cmd+Shift+I (Mac) or Ctrl+Shift+I (Windows) to toggle panel
  useEffect(() => {
    const handleToggleShortcut = (e: KeyboardEvent) => {
      // Check for Cmd+Shift+I on Mac or Ctrl+Shift+I on Windows
      if (
        (e.key === "I" || e.key === "i") &&
        e.shiftKey &&
        (e.metaKey || e.ctrlKey)
      ) {
        e.preventDefault();
        e.stopPropagation();
        onToggle();
      }
    };
    window.addEventListener("keydown", handleToggleShortcut, true); // Use capture phase
    return () =>
      window.removeEventListener("keydown", handleToggleShortcut, true);
  }, [onToggle]);

  return (
    <div className="relative w-full h-full flex overflow-hidden">
      {/* Main content - always rendered, never remounted */}
      <div
        className="h-full"
        style={{
          width: isOpen ? `${100 - panelWidth}%` : "100%",
        }}
      >
        {children}
      </div>

      {/* Agent panel - slides in from right */}
      {isOpen && (
        <div
          className="bg-white dark:bg-zinc-950 border-l border-border h-full flex flex-col"
          style={{
            width: `${panelWidth}%`,
          }}
          ref={panelRef}
        >
          {/* Agent content area */}
          <div className="h-full overflow-hidden pt-1">
            {/* Only render Agent when opened at least once */}
            {hasBeenOpened && (
              <Agent
                traceIds={selectedTraceIds}
                spanIds={selectedSpanIds}
                queryStartTime={traceQueryStartTime}
                queryEndTime={traceQueryEndTime}
                onSpanSelect={selectAgentSpan}
                onViewTypeChange={setViewType}
              />
            )}
          </div>
        </div>
      )}

      {/* Resize handle - only visible when open, matches ResizableHandle styling */}
      {isOpen && (
        <div
          className="absolute top-0 bottom-0 w-1 bg-border hover:bg-muted transition-colors cursor-col-resize z-40 focus-visible:outline-none"
          style={{
            right: `${panelWidth}%`,
          }}
          onMouseDown={(e) => {
            e.preventDefault();
            const startX = e.clientX;
            const startWidth = panelWidth;
            const containerWidth = window.innerWidth;

            const handleMouseMove = (e: MouseEvent) => {
              const currentX = e.clientX;
              const deltaX = startX - currentX;
              const deltaPercent = (deltaX / containerWidth) * 100;
              const newWidth = startWidth + deltaPercent;
              const clampedWidth = Math.min(
                MAX_WIDTH,
                Math.max(MIN_WIDTH, newWidth),
              );
              setPanelWidth(clampedWidth);
            };

            const handleMouseUp = () => {
              document.removeEventListener("mousemove", handleMouseMove);
              document.removeEventListener("mouseup", handleMouseUp);
            };

            document.addEventListener("mousemove", handleMouseMove);
            document.addEventListener("mouseup", handleMouseUp);
          }}
        />
      )}
    </div>
  );
}
