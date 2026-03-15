/**
 * Unit tests for <TimeButton /> and <RefreshButton />
 *
 * TimeButton tests:
 * - Renders with selected time range label
 * - Opens dropdown on click
 * - Calls onTimeRangeSelect when preset is selected
 * - Opens custom dialog when "Custom" is clicked
 * - Disabled state
 *
 * RefreshButton tests:
 * - Renders refresh icon button
 * - Calls onRefresh when clicked
 * - Disabled state
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import { TimeButton, TIME_RANGES } from "@/components/explore/TimeButton";
import RefreshButton from "@/components/explore/RefreshButton";

// ── Mocks ────────────────────────────────────────────────────

// Mock CustomTimeRangeDialog to simplify tests
jest.mock("@/components/explore/CustomTimeRangeDialog", () => ({
  CustomTimeRangeDialog: ({ open, onOpenChange }: any) =>
    open ? <div data-testid="custom-dialog">Custom Dialog</div> : null,
  __esModule: true,
}));

// Mock Tooltip
jest.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => <>{children}</>,
  TooltipTrigger: ({ children }: any) => <>{children}</>,
  TooltipContent: ({ children }: any) => <span>{children}</span>,
}));

// ── TimeButton Tests ─────────────────────────────────────────

describe("<TimeButton />", () => {
  const defaultProps = {
    selectedTimeRange: TIME_RANGES[0], // "Last 1 Minute"
    onTimeRangeSelect: jest.fn(),
    onCustomTimeRangeSelect: jest.fn(),
    currentTimezone: "utc" as const,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders with the selected time range label", () => {
    render(<TimeButton {...defaultProps} />);

    // On large screens, the full label is shown
    expect(screen.getByText("Last 1 Minute")).toBeInTheDocument();
  });

  it("renders as disabled when disabled prop is true", () => {
    render(<TimeButton {...defaultProps} disabled />);

    const trigger = screen.getByRole("button");
    expect(trigger).toBeDisabled();
  });

  it("renders different time range label", () => {
    render(
      <TimeButton
        {...defaultProps}
        selectedTimeRange={{ label: "Last 3 Hours", minutes: 180 }}
      />,
    );

    expect(screen.getByText("Last 3 Hours")).toBeInTheDocument();
  });

  it("shows all preset time ranges in dropdown", async () => {
    const user = userEvent.setup();

    render(<TimeButton {...defaultProps} />);

    await user.click(screen.getByRole("button"));

    // Dropdown should show all TIME_RANGES
    for (const range of TIME_RANGES) {
      const elements = screen.getAllByText(range.label);
      expect(elements.length).toBeGreaterThanOrEqual(1);
    }

    // Should also show "Custom" option
    expect(screen.getByText("Custom")).toBeInTheDocument();
  });

  it("calls onTimeRangeSelect when a preset is clicked", async () => {
    const user = userEvent.setup();

    render(<TimeButton {...defaultProps} />);

    // Open dropdown
    await user.click(screen.getByRole("button"));

    // Click "Last 3 Hours"
    const option = screen.getByText("Last 3 Hours");
    await user.click(option);

    expect(defaultProps.onTimeRangeSelect).toHaveBeenCalledWith(
      expect.objectContaining({
        label: "Last 3 Hours",
        minutes: 180,
      }),
    );
  });

  it("opens custom dialog when Custom option is clicked", async () => {
    const user = userEvent.setup();

    render(<TimeButton {...defaultProps} />);

    // Open dropdown
    await user.click(screen.getByRole("button"));

    // Click "Custom"
    const customOption = screen.getByText("Custom");
    await user.click(customOption);

    // Custom dialog should appear
    expect(screen.getByTestId("custom-dialog")).toBeInTheDocument();
  });
});

// ── RefreshButton Tests ──────────────────────────────────────

describe("<RefreshButton />", () => {
  it("renders a button", () => {
    render(<RefreshButton onRefresh={jest.fn()} />);

    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("calls onRefresh when clicked", async () => {
    const user = userEvent.setup();
    const onRefresh = jest.fn();

    render(<RefreshButton onRefresh={onRefresh} />);

    await user.click(screen.getByRole("button"));

    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it("is disabled when disabled prop is true", () => {
    render(<RefreshButton onRefresh={jest.fn()} disabled />);

    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("does not call onRefresh when disabled", async () => {
    const user = userEvent.setup();
    const onRefresh = jest.fn();

    render(<RefreshButton onRefresh={onRefresh} disabled />);

    await user.click(screen.getByRole("button"));

    expect(onRefresh).not.toHaveBeenCalled();
  });
});
