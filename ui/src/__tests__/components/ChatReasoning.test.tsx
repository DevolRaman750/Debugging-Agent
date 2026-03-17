import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import ChatReasoning from "@/components/right-panel/agent/chat-reasoning";

jest.mock("@/hooks/useSafeAuth", () => ({
  useSafeAuth: () => ({
    getToken: jest.fn().mockResolvedValue("token"),
  }),
}));

describe("<ChatReasoning />", () => {
  beforeEach(() => {
    (global as any).fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        success: true,
        data: [
          {
            chunk_id: 2,
            content: "second",
            status: "in_progress",
            timestamp: "2026-03-15T12:00:02.000Z",
          },
          {
            chunk_id: 1,
            content: "first",
            status: "in_progress",
            timestamp: "2026-03-15T12:00:01.000Z",
          },
        ],
      }),
    });
  });

  it("fetches reasoning endpoint while loading and renders sorted chunks", async () => {
    render(<ChatReasoning chatId="chat-1" isLoading={true} afterTimestamp={0} />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/chat/chat-1/reasoning?after=0",
        expect.any(Object),
      );
      expect(screen.getByText("first")).toBeInTheDocument();
      expect(screen.getByText("second")).toBeInTheDocument();
    });
  });

  it("hides itself when loading is finished and no records exist", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ success: true, data: [] }),
    });

    const { container } = render(
      <ChatReasoning chatId="chat-1" isLoading={false} afterTimestamp={0} />,
    );

    await waitFor(() => {
      expect(container).toBeEmptyDOMElement();
    });
  });
});
