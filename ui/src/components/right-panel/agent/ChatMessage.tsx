import React from "react";
import { RiRobot2Line } from "react-icons/ri";
import { FaGithub } from "react-icons/fa";
import { useSafeUser } from "@/hooks/useSafeAuth";
import { Reference, IntelligenceMetadata } from "@/models/chat";
import { Spinner } from "../../ui/shadcn-io/spinner";
import { Button } from "../../ui/button";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "../../ui/hover-card";
import { ChatReasoning } from "./chat-reasoning";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "../../ui/shadcn-io/ai/reasoning";
import { BarChart3, Check, Copy, Zap } from "lucide-react";

interface Message {
  id: string;
  content: string;
  role: "user" | "assistant" | "github" | "statistics";
  timestamp: Date | string;
  references?: Reference[];
  action_type?: string;
  status?: string;
  metadata?: IntelligenceMetadata;
  user_feedback?: "positive" | "negative" | null;
}

interface ChatMessageProps {
  messages: Message[];
  isLoading: boolean;
  userAvatarUrl?: string;
  messagesEndRef: React.RefObject<HTMLDivElement>;
  onSpanSelect?: (spanId: string) => void;
  onViewTypeChange?: (viewType: "log" | "trace") => void;
  chatId?: string | null;
  onSendMessage?: (message: string) => void;
}

const formatTimestamp = (timestamp: Date | string) => {
  const date = typeof timestamp === "string" ? new Date(timestamp) : timestamp;
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  const y = date.getFullYear();
  const m = months[date.getMonth()];
  const d = date.getDate();
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  const s = String(date.getSeconds()).padStart(2, "0");

  const getOrdinalSuffix = (day: number) => {
    if (day >= 11 && day <= 13) return "th";
    switch (day % 10) {
      case 1:
        return "st";
      case 2:
        return "nd";
      case 3:
        return "rd";
      default:
        return "th";
    }
  };

  return `${y} ${m} ${d}${getOrdinalSuffix(d)} ${h}:${min}:${s}`;
};

const confidenceClass = (confidence: string) => {
  if (confidence === "HIGH") {
    return "bg-emerald-100 text-emerald-700 border-emerald-200";
  }
  if (confidence === "MEDIUM") {
    return "bg-orange-100 text-orange-700 border-orange-200";
  }
  return "bg-red-100 text-red-700 border-red-200";
};

const highlightLogLevel = (level: string) => {
  if (level === "CRITICAL" || level === "ERROR") {
    return "text-red-500 font-bold";
  }
  if (level === "WARNING") {
    return "text-orange-500 font-bold";
  }
  return "";
};

const renderMarkdown = (
  text: string,
  messageId: string,
  references: Reference[] | undefined,
  openHoverCard: string | null,
  setOpenHoverCard: (id: string | null) => void,
  onSpanSelect?: (spanId: string) => void,
  onViewTypeChange?: (viewType: "log" | "trace") => void,
  copiedCodeId?: string | null,
  onCopyCode?: (blockId: string, content: string) => void,
): React.ReactNode => {
  let currentIndex = 0;

  const patterns = [
    {
      regex: /```(\w+)?\n?([\s\S]*?)```/g,
      component: (match: string, language: string, code: string) => {
        const trimmed = (code || "").trim();
        if (!trimmed) {
          return <span key={`${messageId}-empty-code-${currentIndex++}`}>{match}</span>;
        }

        const blockId = `${messageId}-code-${currentIndex}`;
        const copied = copiedCodeId === blockId;

        return (
          <pre
            key={`${messageId}-code-pre-${currentIndex++}`}
            className="bg-zinc-100 dark:bg-zinc-800 border border-zinc-300 dark:border-zinc-700 rounded-md p-3 my-2 overflow-x-auto relative"
          >
            <button
              type="button"
              aria-label="Copy code"
              onClick={() => onCopyCode?.(blockId, trimmed)}
              className="absolute top-2 right-2 inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-900 hover:bg-zinc-50 dark:hover:bg-zinc-800"
            >
              {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copied ? "Copied!" : "Copy"}
            </button>
            {language ? (
              <div className="text-[10px] text-zinc-500 dark:text-zinc-400 mb-2 uppercase tracking-wide">
                {language}
              </div>
            ) : null}
            <code className="text-xs font-mono text-zinc-800 dark:text-zinc-200 whitespace-pre">
              {trimmed}
            </code>
          </pre>
        );
      },
    },
    {
      regex: /\[([^\]]+)\]\(([^)]+)\)/g,
      component: (match: string, label: string, url: string) => (
        <a
          key={`${messageId}-link-${currentIndex++}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-blue-400 underline hover:opacity-80"
        >
          {label}
        </a>
      ),
    },
    {
      regex: /\[(\d+)\]/g,
      component: (match: string, refNumberText: string) => {
        const refNumber = Number(refNumberText);
        const reference = references?.find((item) => item.number === refNumber);
        const hoverId = `${messageId}-ref-${refNumber}-${currentIndex}`;

        if (!reference) {
          return (
            <span
              key={`${messageId}-ref-missing-${currentIndex++}`}
              className="inline-flex items-center rounded-full bg-blue-100 text-blue-700 px-2 py-0.5 text-[11px] font-semibold"
            >
              [{refNumber}]
            </span>
          );
        }

        return (
          <HoverCard
            key={`${messageId}-ref-${currentIndex++}`}
            open={openHoverCard === hoverId}
            onOpenChange={(open) => setOpenHoverCard(open ? hoverId : null)}
          >
            <HoverCardTrigger asChild>
              <button
                type="button"
                className="inline-flex items-center rounded-full bg-blue-100 text-blue-700 hover:bg-blue-200 px-2 py-0.5 text-[11px] font-semibold mx-0.5"
                onClick={() => {
                  if (reference.span_id) {
                    onSpanSelect?.(reference.span_id);
                    onViewTypeChange?.("log");
                  }
                }}
              >
                [{refNumber}]
              </button>
            </HoverCardTrigger>
            <HoverCardContent className="w-96">
              <div className="space-y-2 text-xs">
                {reference.span_id ? (
                  <div>
                    <span className="font-semibold">span_id:</span> {reference.span_id}
                  </div>
                ) : null}
                {reference.span_function_name ? (
                  <div>
                    <span className="font-semibold">span_function_name:</span>{" "}
                    {reference.span_function_name}
                  </div>
                ) : null}
                {reference.line_number ? (
                  <div>
                    <span className="font-semibold">line_number:</span> {reference.line_number}
                  </div>
                ) : null}
                {reference.log_message ? (
                  <div>
                    <span className="font-semibold">log_message:</span>
                    <div className="mt-1 p-2 rounded bg-zinc-100 dark:bg-zinc-800 font-mono whitespace-pre-wrap">
                      {reference.log_message}
                    </div>
                  </div>
                ) : null}
              </div>
            </HoverCardContent>
          </HoverCard>
        );
      },
    },
    {
      regex: /\b(CRITICAL|ERROR|WARNING)\b/g,
      component: (match: string, level: string) => (
        <span key={`${messageId}-level-${currentIndex++}`} className={highlightLogLevel(level)}>
          {level}
        </span>
      ),
    },
    {
      regex: /###\s+([^\n]+)/g,
      component: (match: string, content: string) => (
        <h3 key={`${messageId}-h3-${currentIndex++}`} className="text-sm font-semibold my-1">
          {renderMarkdown(
            content,
            messageId,
            references,
            openHoverCard,
            setOpenHoverCard,
            onSpanSelect,
            onViewTypeChange,
            copiedCodeId,
            onCopyCode,
          )}
        </h3>
      ),
    },
    {
      regex: /\*\*(.*?)\*\*/g,
      component: (match: string, content: string) => (
        <strong key={`${messageId}-bold-${currentIndex++}`}>
          {renderMarkdown(
            content,
            messageId,
            references,
            openHoverCard,
            setOpenHoverCard,
            onSpanSelect,
            onViewTypeChange,
            copiedCodeId,
            onCopyCode,
          )}
        </strong>
      ),
    },
    {
      regex: /\*(.*?)\*/g,
      component: (match: string, content: string) => (
        <em key={`${messageId}-italic-${currentIndex++}`}>
          {renderMarkdown(
            content,
            messageId,
            references,
            openHoverCard,
            setOpenHoverCard,
            onSpanSelect,
            onViewTypeChange,
            copiedCodeId,
            onCopyCode,
          )}
        </em>
      ),
    },
    {
      regex: /`(.*?)`/g,
      component: (match: string, code: string) => (
        <code
          key={`${messageId}-inline-code-${currentIndex++}`}
          className="bg-zinc-100 dark:bg-zinc-700 px-1 py-0.5 rounded text-[11px] font-mono"
        >
          {code}
        </code>
      ),
    },
  ];

  let remaining = text;
  const elements: React.ReactNode[] = [];

  while (remaining.length > 0) {
    let earliestMatch: RegExpExecArray | null = null;
    let earliestIndex = remaining.length;
    let matchedPattern:
      | {
          regex: RegExp;
          component: (...args: string[]) => React.ReactNode;
        }
      | null = null;

    for (const pattern of patterns) {
      const match = pattern.regex.exec(remaining);
      if (match && match.index < earliestIndex) {
        earliestMatch = match;
        earliestIndex = match.index;
        matchedPattern = pattern;
      }
    }

    if (earliestMatch && matchedPattern) {
      if (earliestIndex > 0) {
        elements.push(remaining.substring(0, earliestIndex));
      }

      elements.push(matchedPattern.component(earliestMatch[0], ...earliestMatch.slice(1)));
      remaining = remaining.substring(earliestIndex + earliestMatch[0].length);
      patterns.forEach((pattern) => {
        pattern.regex.lastIndex = 0;
      });
    } else {
      elements.push(remaining);
      break;
    }
  }

  return elements.length > 0 ? elements : text;
};

export default function ChatMessage({
  messages,
  isLoading,
  userAvatarUrl,
  messagesEndRef,
  onSpanSelect,
  onViewTypeChange,
  chatId,
  onSendMessage,
}: ChatMessageProps) {
  const { user } = useSafeUser();
  const avatarLetter =
    user?.emailAddresses?.[0]?.emailAddress?.charAt(0)?.toUpperCase() || "U";
  const [openHoverCard, setOpenHoverCard] = React.useState<string | null>(null);
  const [copiedCodeId, setCopiedCodeId] = React.useState<string | null>(null);

  const handleCopyCode = async (blockId: string, content: string) => {
    try {
      await navigator.clipboard.writeText(content);
      setCopiedCodeId(blockId);
      setTimeout(() => setCopiedCodeId(null), 1200);
    } catch (error) {
      console.error("Failed to copy code block:", error);
    }
  };

  const handleConfirmAction = (value: "yes" | "no") => {
    onSendMessage?.(value);
  };

  return (
    <div className="flex-1 overflow-y-auto p-3 flex flex-col bg-zinc-50 dark:bg-zinc-900 min-h-0">
      <div className="flex-1" />
      {messages.map((message, index) => {
        const shouldShowReasoning = message.role === "user";
        const nextAssistantMessage = messages
          .slice(index + 1)
          .find((item) => item.role === "assistant");
        const nextMessageTimestamp = nextAssistantMessage?.timestamp;
        const isThisMessageLoading = isLoading && !nextAssistantMessage;

        const isUser = message.role === "user";
        const isStatistics = message.role === "statistics";

        const bubbleClass = isUser
          ? "bg-slate-100 dark:bg-slate-800 text-zinc-900 dark:text-zinc-100 border border-slate-200 dark:border-slate-700"
          : "bg-white dark:bg-zinc-900 text-zinc-900 dark:text-zinc-100 border border-zinc-200 dark:border-zinc-700";

        return (
          <React.Fragment key={message.id}>
            <div
              className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3 items-start gap-2`}
            >
              {!isUser ? (
                <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 bg-zinc-700 dark:bg-zinc-200 border border-zinc-700 dark:border-zinc-200">
                  {message.role === "github" ? (
                    <FaGithub className="w-4 h-4 text-white dark:text-zinc-700" />
                  ) : isStatistics ? (
                    <BarChart3 className="w-4 h-4 text-white dark:text-zinc-700" />
                  ) : (
                    <RiRobot2Line className="w-4 h-4 text-white dark:text-zinc-700" />
                  )}
                </div>
              ) : null}

              <div className={`max-w-[72%] rounded-xl px-4 py-3 break-words ${bubbleClass}`}>
                {message.role === "assistant" && message.metadata ? (
                  <div className="mb-2 flex flex-wrap gap-1.5 items-center text-[11px]">
                    <span
                      className={`inline-flex items-center rounded-full border px-2 py-0.5 font-semibold ${confidenceClass(
                        message.metadata.confidence,
                      )}`}
                    >
                      {message.metadata.confidence}
                    </span>
                    {message.metadata.pattern_matched ? (
                      <span className="inline-flex items-center rounded-full border border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5">
                        {message.metadata.pattern_matched}
                      </span>
                    ) : null}
                    {message.metadata.fast_path ? (
                      <span className="inline-flex items-center gap-1 rounded-full border border-yellow-200 bg-yellow-100 text-yellow-700 px-2 py-0.5">
                        <Zap className="w-3 h-3" />
                        Fast Path
                      </span>
                    ) : null}
                    <span className="inline-flex items-center rounded-full border border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5">
                      {message.metadata.processing_time_ms}ms
                    </span>
                  </div>
                ) : null}

                <div className="whitespace-pre-wrap break-words text-xs leading-6">
                  {renderMarkdown(
                    message.content,
                    message.id,
                    message.references,
                    openHoverCard,
                    setOpenHoverCard,
                    onSpanSelect,
                    onViewTypeChange,
                    copiedCodeId,
                    handleCopyCode,
                  )}
                </div>

                <p className="text-[11px] mt-2 opacity-70">
                  {formatTimestamp(message.timestamp)}
                </p>

                {message.action_type === "pending_confirmation" &&
                  message.status === "awaiting_confirmation" && (
                    <div className="mt-3 flex gap-2">
                      <Button
                        type="button"
                        onClick={() => handleConfirmAction("yes")}
                        disabled={isLoading}
                        size="sm"
                        className="bg-green-600 hover:bg-green-700 text-white"
                      >
                        Yes, proceed
                      </Button>
                      <Button
                        type="button"
                        onClick={() => handleConfirmAction("no")}
                        disabled={isLoading}
                        size="sm"
                        variant="outline"
                        className="border-red-500 text-red-600 hover:bg-red-50 dark:hover:bg-red-950"
                      >
                        No, cancel
                      </Button>
                    </div>
                  )}
              </div>

              {isUser ? (
                <div className="w-8 h-8 rounded-full overflow-hidden flex-shrink-0">
                  {userAvatarUrl ? (
                    <img
                      src={userAvatarUrl}
                      alt="User avatar"
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                      <span className="font-semibold text-xs text-slate-800 dark:text-slate-200">
                        {avatarLetter}
                      </span>
                    </div>
                  )}
                </div>
              ) : null}
            </div>

            {shouldShowReasoning ? (
              <div className="mb-3">
                {chatId ? (
                  <ChatReasoning
                    chatId={chatId}
                    className="w-full"
                    isLoading={isThisMessageLoading}
                    userMessageTimestamp={message.timestamp}
                    nextMessageTimestamp={nextMessageTimestamp}
                  />
                ) : (
                  <Reasoning
                    className="w-full"
                    isStreaming={isThisMessageLoading}
                    defaultOpen
                  >
                    <ReasoningTrigger />
                    <ReasoningContent>Processing...</ReasoningContent>
                  </Reasoning>
                )}
              </div>
            ) : null}
          </React.Fragment>
        );
      })}

      {isLoading ? (
        <div className="flex justify-start mb-3 items-start gap-2">
          <div className="w-8 h-8 rounded-full bg-zinc-700 dark:bg-zinc-200 flex items-center justify-center flex-shrink-0 animate-pulse">
            <RiRobot2Line className="w-4 h-4 text-white dark:text-zinc-700" />
          </div>
          <div className="flex items-center justify-center py-1 px-2">
            <Spinner variant="infinite" className="w-7 h-7 text-zinc-500 dark:text-zinc-400" />
          </div>
        </div>
      ) : null}

      <div ref={messagesEndRef} />
    </div>
  );
}
