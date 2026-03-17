import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import FeedbackAction from "@/components/right-panel/agent/FeedbackAction";

describe("<FeedbackAction />", () => {
  beforeEach(() => {
    (global as any).fetch = jest.fn().mockResolvedValue({ ok: true });
  });

  it("submits positive feedback and disables buttons while request is in progress", async () => {
    let resolveRequest: ((value: { ok: boolean }) => void) | undefined;
    (global.fetch as jest.Mock).mockImplementation(
      () =>
        new Promise<{ ok: boolean }>((resolve) => {
          resolveRequest = resolve;
        }),
    );

    render(
      <FeedbackAction
        chatId="chat-42"
        messageTimestamp={1710000000000}
        initialFeedback={null}
      />,
    );

    const upButton = screen.getByRole("button", { name: "Thumbs up" });
    const downButton = screen.getByRole("button", { name: "Thumbs down" });

    fireEvent.click(upButton);

    expect(global.fetch).toHaveBeenCalledWith(
      "/api/feedback",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          chat_id: "chat-42",
          message_timestamp: 1710000000000,
          feedback: "positive",
        }),
      }),
    );

    expect(upButton).toBeDisabled();
    expect(downButton).toBeDisabled();

    expect(resolveRequest).toBeDefined();
    resolveRequest?.({ ok: true });

    await waitFor(() => {
      expect(upButton).not.toBeDisabled();
      expect(downButton).not.toBeDisabled();
    });
  });

  it("renders initial negative feedback as selected", () => {

    render(
      <FeedbackAction
        chatId="chat-99"
        messageTimestamp={1710000000001}
        initialFeedback="negative"
      />,
    );

    const downButton = screen.getByRole("button", { name: "Thumbs down" });
    expect(downButton.className).toContain("text-red-700");
  });
});
