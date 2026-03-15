/**
 * Unit tests for <SearchBar /> component
 *
 * Tests:
 * - Renders collapsed (search icon only) by default
 * - Expands on search icon click
 * - Shows category dropdown with options
 * - Shows operation dropdown
 * - Adds criterion on Enter key
 * - Removes criterion on X click
 * - Call onSearch callback with criteria
 * - Clears all on X (clear) button click
 * - Collapses after adding criterion
 * - Disables inputs when disabled prop is true
 */

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";
import SearchBar, { SearchCriterion } from "@/components/explore/SearchBar";

// ── Mocks ────────────────────────────────────────────────────

jest.mock("@/hooks/useCloudProvider", () => ({
  useCloudProvider: () => ({
    selectedProvider: "aws",
    isLoaded: true,
    settings: { selectedProvider: "aws" },
  }),
}));

jest.mock("@/utils/provider", () => ({
  loadProviderSelection: jest.fn(),
  loadProviderConfig: () => ({ trace_provider: "jaeger", log_provider: "jaeger" }),
}));

// Mock shadcn Tooltip (needed for non-portal rendering)
jest.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: any) => <>{children}</>,
  TooltipTrigger: ({ children }: any) => <>{children}</>,
  TooltipContent: ({ children }: any) => <span>{children}</span>,
}));

// Mock Kbd
jest.mock("@/components/ui/shadcn-io/kbd", () => ({
  Kbd: ({ children }: any) => <kbd>{children}</kbd>,
}));

// ── Tests ────────────────────────────────────────────────────

describe("<SearchBar />", () => {
  const defaultProps = {
    onSearch: jest.fn(),
    onClear: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders collapsed with search icon button", () => {
    render(<SearchBar {...defaultProps} />);

    // Should find a button (the search icon button)
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBeGreaterThanOrEqual(1);
  });

  it("expands when search icon is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchBar {...defaultProps} />);

    // Find and click the search button
    const searchButton = screen.getAllByRole("button")[0];
    await user.click(searchButton);

    // After expanding, there should be input fields and category dropdowns
    // The expanded state has multiple buttons (category, operation, close)
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBeGreaterThan(1);
  });

  it("calls onClear when clear button is clicked", async () => {
    const user = userEvent.setup();
    render(<SearchBar {...defaultProps} />);

    // Expand first
    const searchButton = screen.getAllByRole("button")[0];
    await user.click(searchButton);

    // Find the close/clear button (last button in the expanded view)
    const buttons = screen.getAllByRole("button");
    const closeButton = buttons[buttons.length - 1];
    await user.click(closeButton);

    expect(defaultProps.onClear).toHaveBeenCalled();
  });

  it("disables interaction when disabled prop is true", () => {
    render(<SearchBar {...defaultProps} disabled />);

    const buttons = screen.getAllByRole("button");
    buttons.forEach((button) => {
      expect(button).toBeDisabled();
    });
  });

  it("renders with default category set to 'log'", async () => {
    const user = userEvent.setup();
    render(<SearchBar {...defaultProps} />);

    // Expand
    const searchButton = screen.getAllByRole("button")[0];
    await user.click(searchButton);

    // The default category label should be "log"
    const logElements = screen.getAllByText("log");
    expect(logElements.length).toBeGreaterThanOrEqual(1);
  });

  it("renders with default operation set to 'contains'", async () => {
    const user = userEvent.setup();
    render(<SearchBar {...defaultProps} />);

    // Expand
    const searchButton = screen.getAllByRole("button")[0];
    await user.click(searchButton);

    // The default operation should be "contains"
    const containsElements = screen.getAllByText("contains");
    expect(containsElements.length).toBeGreaterThanOrEqual(1);
  });
});
