import React from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { Button } from "@/components/ui/button";

interface FeedbackActionProps {
  chatId: string;
  messageTimestamp: number;
  initialFeedback?: "positive" | "negative" | null;
}

export default function FeedbackAction({
  chatId,
  messageTimestamp,
  initialFeedback = null,
}: FeedbackActionProps) {
  const [feedback, setFeedback] = React.useState<"positive" | "negative" | null>(
    initialFeedback,
  );
  const [isSubmitting, setIsSubmitting] = React.useState(false);

  const handleSubmit = async (nextFeedback: "positive" | "negative") => {
    setFeedback(nextFeedback);
    setIsSubmitting(true);

    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          chat_id: chatId,
          message_timestamp: messageTimestamp,
          feedback: nextFeedback,
        }),
      });
    } catch (error) {
      console.error("Failed to submit feedback:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const positiveActive = feedback === "positive";
  const negativeActive = feedback === "negative";

  return (
    <div className="flex gap-1 mt-2 justify-end">
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={isSubmitting}
        aria-label="Thumbs up"
        className={`h-7 px-2 ${positiveActive ? "text-emerald-700 bg-emerald-100 hover:bg-emerald-100" : "text-zinc-500 hover:text-emerald-700"}`}
        onClick={() => handleSubmit("positive")}
      >
        <ThumbsUp className="w-3.5 h-3.5" />
      </Button>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={isSubmitting}
        aria-label="Thumbs down"
        className={`h-7 px-2 ${negativeActive ? "text-red-700 bg-red-100 hover:bg-red-100" : "text-zinc-500 hover:text-red-700"}`}
        onClick={() => handleSubmit("negative")}
      >
        <ThumbsDown className="w-3.5 h-3.5" />
      </Button>
    </div>
  );
}
