"use client";

import React, { useEffect, useMemo, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { useSafeAuth } from "@/hooks/useSafeAuth";

interface ChatReasoningProps {
  chatId: string;
  isLoading: boolean;
  afterTimestamp?: number;
}

interface ReasoningRecord {
  chunk_id: number;
  content: string;
  status: "in_progress" | "complete";
  timestamp: string;
}

export default function ChatReasoning({
  chatId,
  isLoading,
  afterTimestamp,
}: ChatReasoningProps) {
  const { getToken } = useSafeAuth();
  const [isOpen, setIsOpen] = useState<boolean>(isLoading);
  const [records, setRecords] = useState<ReasoningRecord[]>([]);

  useEffect(() => {
    if (isLoading) {
      setIsOpen(true);
    }
  }, [isLoading]);

  useEffect(() => {
    let mounted = true;
    let interval: NodeJS.Timeout | undefined;
    let inFlight = false;

    const fetchReasoning = async () => {
      if (!chatId) {
        return;
      }

      if (inFlight) {
        return;
      }

      inFlight = true;

      try {
        const token = await getToken();
        const response = await fetch(
          `/api/chat/${encodeURIComponent(chatId)}/reasoning?after=${afterTimestamp || 0}`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            cache: "no-store",
          },
        );

        if (!response.ok) {
          return;
        }

        const data = await response.json();
        const list: ReasoningRecord[] = Array.isArray(data?.data)
          ? data.data
          : Array.isArray(data?.reasoning)
            ? data.reasoning
            : [];

        if (!mounted) {
          return;
        }

        const sorted = [...list].sort((a, b) => a.chunk_id - b.chunk_id);
        setRecords(sorted);
      } catch (error) {
        console.error("Failed to fetch reasoning chunks:", error);
      } finally {
        inFlight = false;
      }
    };

    void fetchReasoning();

    if (isLoading) {
      interval = setInterval(() => {
        void fetchReasoning();
      }, 1500);
    }

    return () => {
      mounted = false;
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [chatId, isLoading, afterTimestamp]);

  const hasNoData = useMemo(() => records.length === 0, [records]);

  if (!isLoading && hasNoData) {
    return null;
  }

  return (
    <div className="mt-2">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger className="w-full text-left text-xs font-medium text-zinc-600 dark:text-zinc-300 hover:text-zinc-800 dark:hover:text-zinc-100 transition-colors">
          <span className="inline-flex items-center gap-1.5">
            Thinking Process
            {isLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : null}
          </span>
        </CollapsibleTrigger>
        <CollapsibleContent className="mt-2">
          <div className="bg-muted/50 p-3 rounded-md border text-sm text-muted-foreground font-mono whitespace-pre-wrap space-y-2">
            {records.length === 0 ? (
              <div className="text-xs opacity-75">Collecting reasoning chunks...</div>
            ) : (
              records.map((record) => (
                <div key={`${record.chunk_id}-${record.timestamp}`}>{record.content}</div>
              ))
            )}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
