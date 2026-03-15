/**
 * Unit tests for <ExploreHeader />
 *
 * Tests:
 * - Renders SearchBar, RefreshButton, TimeButton, ModeToggle
 * - Sub-components receive context values
 * - Refresh button click triggers refresh
 */

import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import ExploreHeader from "@/components/explore/ExploreHeader";
import { ExploreProvider } from "@/hooks/useExploreContext";

// ── Mocks ────────────────────────────────────────────────────

jest.mock("@/components/explore/SearchBar", () => {
  return function MockSearchBar(props: any) {
    return (
      <div data-testid="search-bar" data-disabled={props.disabled}>
        SearchBar
      </div>
    );
  };
});

jest.mock("@/components/explore/RefreshButton", () => {
  return function MockRefreshButton(props: any) {
    return (
      <button
        data-testid="refresh-button"
        onClick={props.onRefresh}
        disabled={props.disabled}
      >
        Refresh
      </button>
    );
  };
});

jest.mock("@/components/explore/TimeButton", () => ({
  __esModule: true,
  default: function MockTimeButton(props: any) {
    return (
      <button
        data-testid="time-button"
        disabled={props.disabled}
      >
        {props.selectedTimeRange.label}
      </button>
    );
  },
  TimeButton: function MockTimeButton(props: any) {
    return (
      <button
        data-testid="time-button"
        disabled={props.disabled}
      >
        {props.selectedTimeRange.label}
      </button>
    );
  },
  TIME_RANGES: [
    { label: "Last 1 Minute", minutes: 1 },
    { label: "Last 10 Minutes", minutes: 10 },
    { label: "Last 30 Minutes", minutes: 30 },
    { label: "Last 1 Hour", minutes: 60 },
    { label: "Last 3 Hours", minutes: 180 },
  ],
}));

jest.mock("@/components/right-panel/ModeToggle", () => ({
  __esModule: true,
  default: function MockModeToggle(props: any) {
    return <div data-testid="mode-toggle">{props.viewType}</div>;
  },
}));

jest.mock("@/components/explore/CustomTimeRangeDialog", () => ({
  CustomTimeRangeDialog: () => null,
}));

// ── Helper ───────────────────────────────────────────────────

function renderWithProvider() {
  return render(
    <ExploreProvider>
      <ExploreHeader />
    </ExploreProvider>,
  );
}

// ── Tests ────────────────────────────────────────────────────

describe("<ExploreHeader />", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders all sub-components", () => {
    renderWithProvider();

    expect(screen.getByTestId("search-bar")).toBeInTheDocument();
    expect(screen.getByTestId("refresh-button")).toBeInTheDocument();
    expect(screen.getByTestId("time-button")).toBeInTheDocument();
    expect(screen.getByTestId("mode-toggle")).toBeInTheDocument();
  });

  it("displays default time range from context", () => {
    renderWithProvider();

    expect(screen.getByTestId("time-button")).toHaveTextContent("Last 3 Hours");
  });

  it("displays default viewType from context", () => {
    renderWithProvider();

    expect(screen.getByTestId("mode-toggle")).toHaveTextContent("trace");
  });

  it("triggers refresh when RefreshButton is clicked", async () => {
    const user = userEvent.setup();
    renderWithProvider();

    // Clicking refresh should not throw — the handler comes from context
    await user.click(screen.getByTestId("refresh-button"));
  });

  it("search bar is not disabled by default", () => {
    renderWithProvider();

    expect(screen.getByTestId("search-bar")).toHaveAttribute(
      "data-disabled",
      "false",
    );
  });
});
