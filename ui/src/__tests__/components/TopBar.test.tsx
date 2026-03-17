import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import TopBar from "@/components/right-panel/agent/TopBar";

jest.mock("@/hooks/useSafeAuth", () => ({
  useSafeAuth: () => ({
    getToken: jest.fn().mockResolvedValue("token"),
  }),
}));

describe("<TopBar />", () => {
  beforeEach(() => {
    (global as any).fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        data: {
          history: [
            {
              chat_id: "chat-1",
              chat_title: "Today Chat",
              trace_id: "trace-1",
              timestamp: Date.now(),
            },
          ],
        },
        hasMore: false,
      }),
    });
  });

  it("renders tabs and handles close/select actions", () => {
    const onTabSelect = jest.fn();
    const onTabClose = jest.fn();

    render(
      <TopBar
        tabs={[
          { id: "tab-1", title: "Root Cause", isNew: false },
          { id: "tab-2", title: "Unsaved", isNew: true },
        ]}
        activeTabId="tab-1"
        traceId="trace-1"
        onTabSelect={onTabSelect}
        onTabClose={onTabClose}
        onNewChat={jest.fn()}
        onHistoryItemSelect={jest.fn()}
      />,
    );

    fireEvent.click(screen.getByText("Root Cause"));
    expect(onTabSelect).toHaveBeenCalledWith("tab-1");

    fireEvent.click(screen.getByLabelText("Close Root Cause"));
    expect(onTabClose).toHaveBeenCalledWith("tab-1");

    expect(screen.getByText("New Chat")).toBeInTheDocument();
  });

  it("fetches history on dropdown open and selects history item", async () => {
    const onHistoryItemSelect = jest.fn();

    render(
      <TopBar
        tabs={[{ id: "tab-1", title: "Root Cause", isNew: false }]}
        activeTabId="tab-1"
        traceId="trace-1"
        onTabSelect={jest.fn()}
        onTabClose={jest.fn()}
        onNewChat={jest.fn()}
        onHistoryItemSelect={onHistoryItemSelect}
      />,
    );

    fireEvent.click(screen.getByTitle("History"));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
      expect(screen.getByText("Today")).toBeInTheDocument();
      expect(screen.getByText("Today Chat")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Today Chat"));
    expect(onHistoryItemSelect).toHaveBeenCalledWith("chat-1");
  });
});
