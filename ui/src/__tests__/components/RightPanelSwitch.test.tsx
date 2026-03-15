import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import RightPanelSwitch from "@/components/right-panel/RightPanelSwitch";
import { Trace } from "@/models/trace";

const capturedProps: { log: any[]; trace: any[] } = {
  log: [],
  trace: [],
};

jest.mock("@/components/right-panel/log/LogPanelSwitch", () => ({
  __esModule: true,
  default: function MockLogPanelSwitch(props: any) {
    capturedProps.log.push(props);
    return <div data-testid="log-panel">LogPanel</div>;
  },
}));

jest.mock("@/components/right-panel/trace/TracePanelSwitch", () => ({
  __esModule: true,
  default: function MockTracePanelSwitch(props: any) {
    capturedProps.trace.push(props);
    return <div data-testid="trace-panel">TracePanel</div>;
  },
}));

const traces: Trace[] = [
  {
    id: "trace-1",
    service_name: "svc-a",
    duration: 120,
    start_time: 1000,
    end_time: 1120,
    percentile: "P50",
    spans: [
      { id: "span-a1", name: "A1", start_time: 1000, end_time: 1050, duration: 50 },
      { id: "span-a2", name: "A2", start_time: 1050, end_time: 1120, duration: 70 },
    ],
    telemetry_sdk_language: ["python"],
  },
  {
    id: "trace-2",
    service_name: "svc-b",
    duration: 300,
    start_time: 2000,
    end_time: 2300,
    percentile: "P95",
    spans: [{ id: "span-b1", name: "B1", start_time: 2000, end_time: 2300, duration: 300 }],
    telemetry_sdk_language: ["python"],
  },
];

describe("<RightPanelSwitch />", () => {
  beforeEach(() => {
    capturedProps.log = [];
    capturedProps.trace = [];
  });

  it("routes to LogPanelSwitch when viewType is log", async () => {
    render(
      <RightPanelSwitch
        viewType="log"
        traceIds={["trace-1"]}
        spanIds={[]}
        allTraces={traces}
      />,
    );

    expect(screen.getByTestId("log-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("trace-panel")).not.toBeInTheDocument();
  });

  it("routes to TracePanelSwitch when viewType is trace", async () => {
    render(
      <RightPanelSwitch
        viewType="trace"
        traceIds={["trace-1"]}
        spanIds={[]}
        allTraces={traces}
      />,
    );

    expect(screen.getByTestId("trace-panel")).toBeInTheDocument();
    expect(screen.queryByTestId("log-panel")).not.toBeInTheDocument();
  });

  it("extracts spans from a single selected trace", async () => {
    render(
      <RightPanelSwitch
        viewType="trace"
        traceIds={["trace-1"]}
        spanIds={[]}
        allTraces={traces}
      />,
    );

    await waitFor(() => {
      const traceProps = capturedProps.trace[capturedProps.trace.length - 1];
      const segmentIds = traceProps.segments.map((s: { id: string }) => s.id);

      expect(segmentIds).toEqual(["span-a1", "span-a2"]);
      expect(traceProps.traceId).toBe("trace-1");
    });
  });

  it("merges spans when multiple traces are selected", async () => {
    render(
      <RightPanelSwitch
        viewType="trace"
        traceIds={["trace-1", "trace-2"]}
        spanIds={[]}
        allTraces={traces}
      />,
    );

    await waitFor(() => {
      const traceProps = capturedProps.trace[capturedProps.trace.length - 1];
      const segmentIds = traceProps.segments.map((s: { id: string }) => s.id);

      expect(segmentIds).toEqual(["span-a1", "span-a2", "span-b1"]);
      expect(traceProps.traceId).toBeUndefined();
    });
  });

  it("prepares trace arrays for child views", async () => {
    render(
      <RightPanelSwitch
        viewType="log"
        traceIds={[]}
        spanIds={[]}
        allTraces={traces}
      />,
    );

    await waitFor(() => {
      const logProps = capturedProps.log[capturedProps.log.length - 1];

      expect(logProps.traceDurations).toEqual([120, 300]);
      expect(logProps.traceIDs).toEqual(["trace-1", "trace-2"]);
      expect(logProps.tracePercentiles).toEqual(["P50", "P95"]);
      expect(logProps.traceStartTimes[0]).toEqual(new Date(1000 * 1000));
      expect(logProps.traceEndTimes[1]).toEqual(new Date(2300 * 1000));
    });
  });
});
