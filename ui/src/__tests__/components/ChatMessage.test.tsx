import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import ChatMessage from "@/components/right-panel/agent/ChatMessage";

jest.mock("@/hooks/useSafeAuth", () => ({
  useSafeUser: () => ({
    user: {
      emailAddresses: [{ emailAddress: "dev@example.com" }],
    },
  }),
}));

jest.mock("@/components/right-panel/agent/chat-reasoning", () => ({
  __esModule: true,
  default: () => <div data-testid="chat-reasoning" />,
}));

jest.mock("@/components/ui/shadcn-io/ai/reasoning", () => ({
  __esModule: true,
  Reasoning: ({ children }: any) => <div>{children}</div>,
  ReasoningContent: ({ children }: any) => <div>{children}</div>,
  ReasoningTrigger: () => <div>Reasoning</div>,
}));

describe("<ChatMessage />", () => {
  beforeEach(() => {
    (global as any).fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ success: true }),
    });

    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: {
        writeText: jest.fn().mockResolvedValue(undefined),
      },
    });
  });

  it("renders metadata bar and reference pill, and selects span on click", async () => {
    const onSpanSelect = jest.fn();

    render(
      <ChatMessage
        messages={[
          {
            id: "assistant-1",
            content: "Investigate [1] and ERROR in the logs",
            role: "assistant",
            timestamp: new Date("2026-03-15T10:00:00.000Z"),
            references: [
              {
                number: 1,
                span_id: "span-123",
                span_function_name: "db.query",
                line_number: 88,
                log_message: "timeout",
              },
            ],
            metadata: {
              confidence: "HIGH",
              pattern_matched: "latency_spike",
              fast_path: true,
              processing_time_ms: 321,
              top_cause: "db",
              top_cause_score: 0.9,
              causes_found: 1,
            },
          },
        ]}
        isLoading={false}
        messagesEndRef={{ current: null }}
        onSpanSelect={onSpanSelect}
        chatId="chat-1"
      />,
    );

    expect(screen.getByText("HIGH")).toBeInTheDocument();
    expect(screen.getByText("latency_spike")).toBeInTheDocument();
    expect(screen.getByText("Fast Path")).toBeInTheDocument();
    expect(screen.getByText("321ms")).toBeInTheDocument();

    const refPill = screen.getByRole("button", { name: "[1]" });
    fireEvent.click(refPill);
    expect(onSpanSelect).toHaveBeenCalledWith("span-123");

    expect(screen.getByText("ERROR")).toBeInTheDocument();

    const thumbsUp = screen.getByRole("button", { name: "Thumbs up" });
    fireEvent.click(thumbsUp);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/feedback",
        expect.objectContaining({
          method: "POST",
        }),
      );
    });
  });

  it("copies fenced code and shows temporary copied state", async () => {
    render(
      <ChatMessage
        messages={[
          {
            id: "assistant-2",
            content: "```ts\nconst a = 1;\n```",
            role: "assistant",
            timestamp: new Date("2026-03-15T10:01:00.000Z"),
          },
        ]}
        isLoading={false}
        messagesEndRef={{ current: null }}
      />,
    );

    const copyButton = screen.getByRole("button", { name: "Copy code" });
    fireEvent.click(copyButton);

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith("const a = 1;");
      expect(screen.getByText("Copied!")).toBeInTheDocument();
    });
  });

  it("renders confirmation actions and sends no/yes reply", () => {
    const onSendMessage = jest.fn();

    render(
      <ChatMessage
        messages={[
          {
            id: "assistant-3",
            content: "Need confirmation",
            role: "assistant",
            timestamp: new Date("2026-03-15T10:02:00.000Z"),
            action_type: "pending_confirmation",
            status: "awaiting_confirmation",
          },
        ]}
        isLoading={false}
        messagesEndRef={{ current: null }}
        onSendMessage={onSendMessage}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "No, cancel" }));
    expect(onSendMessage).toHaveBeenCalledWith("no");

    fireEvent.click(screen.getByRole("button", { name: "Yes, proceed" }));
    expect(onSendMessage).toHaveBeenCalledWith("yes");
  });
});
