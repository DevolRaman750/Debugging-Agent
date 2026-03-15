import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import MessageInput from "@/components/right-panel/agent/MessageInput";

jest.mock("@/components/ui/shadcn-io/navbar-13", () => ({
  __esModule: true,
  Navbar13: ({ options, label }: any) => (
    <div data-testid={`navbar-${label}`}>
      {options.map((opt: any) => (
        <span key={String(opt.value)}>{opt.name}</span>
      ))}
    </div>
  ),
}));

describe("<MessageInput />", () => {
  const baseProps = {
    inputMessage: "",
    setInputMessage: jest.fn(),
    isLoading: false,
    onSendMessage: jest.fn(),
    selectedModel: "llama-3.3-70b-versatile" as const,
    setSelectedModel: jest.fn(),
    selectedMode: "agent" as const,
    setSelectedMode: jest.fn(),
    traceId: undefined,
    traceIds: [],
    spanIds: [],
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("shows disabled placeholder when no trace is selected", () => {
    render(<MessageInput {...baseProps} />);

    const textarea = screen.getByPlaceholderText(
      "Select a trace to start chatting",
    );
    expect(textarea).toBeDisabled();

    expect(screen.getByRole("button")).toBeDisabled();
  });

  it("enables send only when trace selected and message is non-empty", () => {
    const { rerender } = render(
      <MessageInput {...baseProps} traceId="trace-123" inputMessage="" />,
    );

    const sendButton = screen.getByRole("button");
    expect(sendButton).toBeDisabled();

    rerender(
      <MessageInput
        {...baseProps}
        traceId="trace-123"
        inputMessage="Investigate this trace"
      />,
    );

    expect(screen.getByRole("button")).not.toBeDisabled();
  });

  it("submits on Enter and keeps newline behavior for Shift+Enter", () => {
    const onSendMessage = jest.fn();

    render(
      <MessageInput
        {...baseProps}
        onSendMessage={onSendMessage}
        traceId="trace-123"
        inputMessage="hello"
      />,
    );

    const textarea = screen.getByPlaceholderText("Type your message...");

    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSendMessage).toHaveBeenCalledTimes(1);

    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSendMessage).toHaveBeenCalledTimes(1);
  });

  it("renders mode options and only current selected model", () => {
    render(<MessageInput {...baseProps} traceId="trace-123" />);

    const modeNavbar = screen.getByTestId("navbar-Mode");
    expect(modeNavbar).toHaveTextContent("Agent 🤖");
    expect(modeNavbar).toHaveTextContent("Chat 💬");

    const modelNavbar = screen.getByTestId("navbar-Model");
    expect(modelNavbar).toHaveTextContent("Llama 3.3 70B");
    expect(modelNavbar).not.toHaveTextContent("GPT-4o");
    expect(modelNavbar).not.toHaveTextContent("GPT-4.1-mini");
    expect(modelNavbar).not.toHaveTextContent("Auto");
  });

  it("shows context badges for trace_id and span count", () => {
    render(
      <MessageInput
        {...baseProps}
        traceId="trace-abcdef123456"
        spanIds={["span-1", "span-2"]}
      />,
    );

    expect(screen.getByText(/trace_id:/i)).toBeInTheDocument();
    expect(screen.getByText("spans: 2")).toBeInTheDocument();
  });

  it("uses auto-resizing textarea baseline with 1 row", () => {
    render(<MessageInput {...baseProps} traceId="trace-123" />);

    const textarea = screen.getByPlaceholderText("Type your message...");
    expect(textarea).toHaveAttribute("rows", "1");
  });
});
