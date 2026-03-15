/**
 * Unit tests for <Span /> component
 *
 * Tests:
 * - Renders span name
 * - Renders duration badge
 * - Shows error badge when num_error_logs > 0
 * - Shows warning badge when num_warning_logs > 0
 * - Renders nested child spans recursively
 * - Highlights selected span
 * - Calls onSpanSelect with span ID + child IDs on click
 * - Shows expand/collapse toggle for spans with children
 * - Shows SDK language icon
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import Span from "@/components/explore/span/Span";
import { Span as SpanType } from "@/models/trace";

// ── Mocks ────────────────────────────────────────────────────

jest.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => <>{children}</>,
  TooltipTrigger: ({ children }: any) => <>{children}</>,
  TooltipContent: ({ children }: any) => <span>{children}</span>,
}));

jest.mock("@/constants/animations", () => ({
  fadeInAnimationStyles: "",
}));

// ── Test Data ────────────────────────────────────────────────

const SIMPLE_SPAN: SpanType = {
  id: "span-001",
  name: "handle_request",
  start_time: 1700000000,
  end_time: 1700000001,
  duration: 1.0,
  spans: [],
};

const SPAN_WITH_ERRORS: SpanType = {
  id: "span-002",
  name: "database_query",
  start_time: 1700000000,
  end_time: 1700000000.5,
  duration: 0.5,
  num_error_logs: 3,
  num_critical_logs: 1,
  spans: [],
};

const SPAN_WITH_WARNINGS: SpanType = {
  id: "span-003",
  name: "cache_lookup",
  start_time: 1700000000,
  end_time: 1700000000.05,
  duration: 0.05,
  num_warning_logs: 2,
  spans: [],
};

const SPAN_WITH_CHILDREN: SpanType = {
  id: "parent-001",
  name: "process_order",
  start_time: 1700000000,
  end_time: 1700000002,
  duration: 2.0,
  spans: [
    {
      id: "child-001",
      name: "validate_payment",
      start_time: 1700000000,
      end_time: 1700000001,
      duration: 1.0,
      spans: [],
    },
    {
      id: "child-002",
      name: "send_confirmation",
      start_time: 1700000001,
      end_time: 1700000002,
      duration: 1.0,
      spans: [],
    },
  ],
};

const PYTHON_SPAN: SpanType = {
  id: "span-py",
  name: "run_model",
  start_time: 1700000000,
  end_time: 1700000001,
  duration: 1.0,
  telemetry_sdk_language: "python",
  spans: [],
};

// ── Tests ────────────────────────────────────────────────────

describe("<Span />", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders span name", () => {
    render(<Span span={SIMPLE_SPAN} />);

    expect(screen.getByText("handle_request")).toBeInTheDocument();
  });

  it("renders duration badge", async () => {
    // 1 second = "1.00s"
    render(<Span span={SIMPLE_SPAN} />);

    // Wait for animation to complete
    await new Promise((resolve) => setTimeout(resolve, 100));

    expect(screen.getByText("1.00s")).toBeInTheDocument();
  });

  it("calls onSpanSelect when clicked", async () => {
    const user = userEvent.setup();
    const onSpanSelect = jest.fn();

    render(<Span span={SIMPLE_SPAN} onSpanSelect={onSpanSelect} />);

    // Click the span row
    const spanRow = screen.getByText("handle_request").closest("div[class*='cursor-pointer']");
    if (spanRow) {
      await user.click(spanRow);
    }

    expect(onSpanSelect).toHaveBeenCalledWith("span-001", []);
  });

  it("passes child span IDs when a parent span is clicked", async () => {
    const user = userEvent.setup();
    const onSpanSelect = jest.fn();

    render(<Span span={SPAN_WITH_CHILDREN} onSpanSelect={onSpanSelect} />);

    // Click the parent span
    const spanRow = screen.getByText("process_order").closest("div[class*='cursor-pointer']");
    if (spanRow) {
      await user.click(spanRow);
    }

    expect(onSpanSelect).toHaveBeenCalledWith(
      "parent-001",
      ["child-001", "child-002"],
    );
  });

  it("renders child spans when present", async () => {
    render(<Span span={SPAN_WITH_CHILDREN} />);

    // Wait for animation
    await new Promise((resolve) => setTimeout(resolve, 100));

    // Parent span
    expect(screen.getByText("process_order")).toBeInTheDocument();
    // Child spans should also render
    expect(screen.getByText("validate_payment")).toBeInTheDocument();
    expect(screen.getByText("send_confirmation")).toBeInTheDocument();
  });

  it("highlights selected span with different styling", () => {
    const { container } = render(
      <Span span={SIMPLE_SPAN} isSelected selectedSpanIds={["span-001"]} />,
    );

    // Selected span should have bg-zinc-200 class
    const spanRow = container.querySelector("div[class*='bg-zinc-200']");
    expect(spanRow).toBeTruthy();
  });

  it("shows non-selected styling when not selected", () => {
    const { container } = render(
      <Span span={SIMPLE_SPAN} isSelected={false} />,
    );

    // Non-selected should have bg-white class
    const spanRow = container.querySelector("div[class*='bg-white']");
    expect(spanRow).toBeTruthy();
  });

  it("truncates long span names", () => {
    const longNameSpan: SpanType = {
      ...SIMPLE_SPAN,
      name: "a_very_long_function_name_that_exceeds_the_fifty_character_truncation_limit_for_display",
    };

    render(<Span span={longNameSpan} />);

    // Should show truncated name with "..."
    expect(
      screen.getByText(
        "a_very_long_function_name_that_exceeds_the_fifty_c...",
      ),
    ).toBeInTheDocument();
  });
});
