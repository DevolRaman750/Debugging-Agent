"use client";

import React, { useMemo, useState } from "react";
import { format, isToday, isYesterday, subDays } from "date-fns";
import { Clock3, Plus, X } from "lucide-react";
import { ChatMetadata, ChatTab } from "@/models/chat";
import { useSafeAuth } from "@/hooks/useSafeAuth";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/shadcn-io/spinner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface TopBarProps {
  tabs: ChatTab[];
  activeTabId: string;
  traceId: string;
  onTabSelect: (id: string) => void;
  onTabClose: (id: string) => void;
  onNewChat: () => void;
  onHistoryItemSelect: (chatId: string) => void;
}

interface GroupedHistory {
  label: "Today" | "Yesterday" | "Last 7 Days" | "Older";
  items: ChatMetadata[];
}

function normalizeHistoryResponse(raw: any): ChatMetadata[] {
  const history: ChatMetadata[] = Array.isArray(raw?.history)
    ? raw.history
    : Array.isArray(raw?.data?.history)
      ? raw.data.history
      : [];

  return history
    .map((item) => ({
      ...item,
      timestamp:
        typeof item.timestamp === "string"
          ? new Date(item.timestamp).getTime()
          : item.timestamp,
    }))
    .sort((a, b) => b.timestamp - a.timestamp);
}

function groupHistoryByDate(items: ChatMetadata[]): GroupedHistory[] {
  const last7DaysCutoff = subDays(new Date(), 7);

  const grouped: GroupedHistory[] = [
    { label: "Today", items: [] },
    { label: "Yesterday", items: [] },
    { label: "Last 7 Days", items: [] },
    { label: "Older", items: [] },
  ];

  items.forEach((item) => {
    const date = new Date(item.timestamp);
    if (isToday(date)) {
      grouped[0].items.push(item);
      return;
    }
    if (isYesterday(date)) {
      grouped[1].items.push(item);
      return;
    }
    if (date >= last7DaysCutoff) {
      grouped[2].items.push(item);
      return;
    }
    grouped[3].items.push(item);
  });

  return grouped.filter((group) => group.items.length > 0);
}

export default function TopBar({
  tabs,
  activeTabId,
  traceId,
  onTabSelect,
  onTabClose,
  onNewChat,
  onHistoryItemSelect,
}: TopBarProps) {
  const { getToken } = useSafeAuth();
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [historyItems, setHistoryItems] = useState<ChatMetadata[]>([]);

  const groupedHistory = useMemo(
    () => groupHistoryByDate(historyItems),
    [historyItems],
  );

  const fetchHistory = async () => {
    if (!traceId) {
      setHistoryItems([]);
      return;
    }

    setIsLoadingHistory(true);
    try {
      const token = await getToken();
      const response = await fetch(
        `/api/get_chat_metadata_history?trace_id=${encodeURIComponent(traceId)}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch metadata history: ${response.status}`);
      }

      const data = await response.json();
      setHistoryItems(normalizeHistoryResponse(data));
    } catch (error) {
      console.error("Failed to load chat metadata history:", error);
      setHistoryItems([]);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  return (
    <div className="flex items-center gap-2 border-b border-neutral-300 dark:border-neutral-700 bg-white dark:bg-black px-2 py-2">
      <div className="flex-1 min-w-0 overflow-x-auto">
        <div className="flex items-center gap-1.5 min-w-max pr-2">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTabId;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => onTabSelect(tab.id)}
                className={`group inline-flex items-center gap-2 h-8 px-2 rounded-md border text-xs transition-colors ${
                  isActive
                    ? "bg-zinc-200 dark:bg-zinc-700 border-zinc-300 dark:border-zinc-600"
                    : "bg-white dark:bg-zinc-900 hover:bg-zinc-100 dark:hover:bg-zinc-800 border-zinc-200 dark:border-zinc-700"
                }`}
              >
                <span className="whitespace-nowrap overflow-hidden text-ellipsis max-w-[150px]">
                  {tab.isNew ? "New Chat" : tab.title}
                </span>
                <span
                  role="button"
                  aria-label={`Close ${tab.title}`}
                  tabIndex={0}
                  className="opacity-70 group-hover:opacity-100"
                  onClick={(e) => {
                    e.stopPropagation();
                    onTabClose(tab.id);
                  }}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      e.stopPropagation();
                      onTabClose(tab.id);
                    }
                  }}
                >
                  <X className="w-3 h-3" />
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <Button
          type="button"
          size="icon"
          variant="outline"
          className="h-8 w-8"
          onClick={onNewChat}
          title="New Chat"
        >
          <Plus className="w-4 h-4" />
        </Button>

        <DropdownMenu
          open={isHistoryOpen}
          onOpenChange={(open) => {
            setIsHistoryOpen(open);
            if (open) {
              void fetchHistory();
            }
          }}
        >
          <DropdownMenuTrigger asChild>
            <Button
              type="button"
              size="icon"
              variant="outline"
              className="h-8 w-8"
              title="History"
              onClick={() => {
                setIsHistoryOpen(true);
                void fetchHistory();
              }}
            >
              <Clock3 className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>

          <DropdownMenuContent align="end" className="w-80 max-h-96 overflow-y-auto">
            {isLoadingHistory ? (
              <div className="flex items-center justify-center py-6">
                <Spinner className="w-5 h-5" />
              </div>
            ) : groupedHistory.length === 0 ? (
              <div className="px-3 py-2 text-xs text-zinc-500">No history found</div>
            ) : (
              groupedHistory.map((group) => (
                <DropdownMenuGroup key={group.label}>
                  <DropdownMenuLabel className="text-[11px] uppercase tracking-wide text-zinc-500">
                    {group.label}
                  </DropdownMenuLabel>
                  {group.items.map((item) => (
                    <DropdownMenuItem
                      key={item.chat_id}
                      onClick={() => {
                        onHistoryItemSelect(item.chat_id);
                        setIsHistoryOpen(false);
                      }}
                      className="flex flex-col items-start gap-0.5"
                    >
                      <span className="text-xs font-medium w-full truncate">
                        {item.chat_title || "Untitled Chat"}
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        {format(new Date(item.timestamp), "MMM d, h:mm a")}
                      </span>
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuGroup>
              ))
            )}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </div>
  );
}
