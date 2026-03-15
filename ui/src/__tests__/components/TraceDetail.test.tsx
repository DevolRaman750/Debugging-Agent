import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TraceDetail from "@/components/right-panel/trace/TraceDetail";
import { Span } from "@/models/trace";

describe("<TraceDetail />", () => {
  const spans: Span[] = [
    {
      id: "root-span",
      name: "api.request",
      start_time: 100,
      end_time: 110,
      duration: 10,
      spans: [
        {
          id: "db-span",
          name: "db.query",
          start_time: 102,
          end_time: 107,
          duration: 5,
          spans: [],
        },
      ],
    },
  ];

  it("positions nested spans relative to global timeline", () => {
    render(
      <TraceDetail
        traceId="trace-1"
        spans={spans}
        spanIds={[]}
        onSpanSelect={jest.fn()}
      />,
    );

    const childBar = screen.getByTitle("db.query - db.query (5.000s)");
    expect(childBar).toHaveStyle({ left: "20%", width: "50%" });

    expect(screen.getByText("Trace Depth Chart")).toBeInTheDocument();
  });

  it("calls onSpanSelect and supports drill-down click", () => {
    const onSpanSelect = jest.fn();

    render(
      <TraceDetail
        traceId="trace-1"
        spans={spans}
        spanIds={[]}
        onSpanSelect={onSpanSelect}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "db.query" }));

    expect(onSpanSelect).toHaveBeenCalledTimes(1);
    expect(onSpanSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        id: "db-span",
      }),
    );
  });
});
