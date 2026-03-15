import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TraceDepthChart from "@/components/right-panel/trace/TraceDepthChart";
import { Span } from "@/models/trace";

interface WaterfallRow {
  span: Span;
  depth: number;
}

describe("<TraceDepthChart />", () => {
  const rows: WaterfallRow[] = [
    {
      span: {
        id: "root",
        name: "root.request",
        start_time: 100,
        end_time: 120,
        duration: 20,
      },
      depth: 0,
    },
    {
      span: {
        id: "child-a",
        name: "child.a",
        start_time: 105,
        end_time: 112,
        duration: 7,
      },
      depth: 1,
    },
    {
      span: {
        id: "child-b",
        name: "child.b",
        start_time: 106,
        end_time: 108,
        duration: 2,
      },
      depth: 2,
    },
    {
      span: {
        id: "sibling",
        name: "sibling.call",
        start_time: 112,
        end_time: 118,
        duration: 6,
      },
      depth: 1,
    },
  ];

  it("renders lanes by nesting depth and complexity summary", () => {
    render(
      <TraceDepthChart rows={rows} timelineStart={100} timelineDuration={20} />,
    );

    expect(screen.getByText("Trace Depth Chart")).toBeInTheDocument();
    expect(screen.getByTestId("depth-lane-0")).toBeInTheDocument();
    expect(screen.getByTestId("depth-lane-1")).toBeInTheDocument();
    expect(screen.getByTestId("depth-lane-2")).toBeInTheDocument();

    const complexity = screen.getByTestId("trace-complexity-indicator");
    expect(complexity).toHaveTextContent("Depth 3 lanes");
    expect(complexity).toHaveTextContent("Peak fan-out 2");
  });

  it("positions bars using the global timeline percentages", () => {
    render(
      <TraceDepthChart rows={rows} timelineStart={100} timelineDuration={20} />,
    );

    const childA = screen.getByTestId("depth-bar-child-a");
    expect(childA).toHaveStyle({ left: "25%", width: "35%" });

    const sibling = screen.getByTestId("depth-bar-sibling");
    expect(sibling).toHaveStyle({ left: "60%", width: "30%" });
  });
});
