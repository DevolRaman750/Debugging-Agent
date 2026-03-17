import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import IntelligenceMetadataBar from "@/components/right-panel/agent/IntelligenceMetadataBar";

describe("<IntelligenceMetadataBar />", () => {
  it("renders confidence, pattern, fast path, processing time, and causes found", () => {
    render(
      <IntelligenceMetadataBar
        metadata={{
          confidence: "HIGH",
          pattern_matched: "N+1 Query",
          fast_path: true,
          processing_time_ms: 412,
          top_cause: null,
          top_cause_score: null,
          causes_found: 3,
        }}
      />,
    );

    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("N+1 Query")).toBeInTheDocument();
    expect(screen.getByText("Fast Path")).toBeInTheDocument();
    expect(screen.getByText("412ms")).toBeInTheDocument();
    expect(screen.getByText("3 causes found")).toBeInTheDocument();
  });

  it("renders deep analysis indicator when fast_path is false", () => {
    render(
      <IntelligenceMetadataBar
        metadata={{
          confidence: "LOW",
          pattern_matched: null,
          fast_path: false,
          processing_time_ms: 99,
          top_cause: null,
          top_cause_score: null,
          causes_found: 1,
        }}
      />,
    );

    expect(screen.getByText("Deep Analysis")).toBeInTheDocument();
    expect(screen.queryByText("causes found")).not.toBeInTheDocument();
  });
});
