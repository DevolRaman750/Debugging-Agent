import { NextRequest, NextResponse } from "next/server";
import { createBackendAuthHeaders } from "@/lib/server-auth-headers";

interface FeedbackBody {
  chat_id: string;
  message_timestamp: number;
  feedback: "positive" | "negative";
}

function isValidFeedbackBody(body: unknown): body is FeedbackBody {
  if (!body || typeof body !== "object") {
    return false;
  }

  const payload = body as Partial<FeedbackBody>;
  const hasValidChatId =
    typeof payload.chat_id === "string" && payload.chat_id.trim().length > 0;
  const hasValidTimestamp =
    typeof payload.message_timestamp === "number" &&
    Number.isFinite(payload.message_timestamp);
  const hasValidFeedback =
    payload.feedback === "positive" || payload.feedback === "negative";

  return hasValidChatId && hasValidTimestamp && hasValidFeedback;
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  try {
    const body: unknown = await req.json();

    if (!isValidFeedbackBody(body)) {
      return NextResponse.json(
        {
          success: false,
          error:
            "Invalid request body. Expected chat_id, message_timestamp, and feedback (positive|negative).",
        },
        { status: 400 },
      );
    }

    const restApiEndpoint = process.env.REST_API_ENDPOINT;
    if (!restApiEndpoint) {
      throw new Error("REST_API_ENDPOINT is not configured");
    }

    const apiUrl = `${restApiEndpoint}/v1/explore/update-feedback`;
    const headers = await createBackendAuthHeaders(req);
    const backendRes = await fetch(apiUrl, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });

    let backendPayload: unknown = null;
    try {
      backendPayload = await backendRes.json();
    } catch {
      backendPayload = null;
    }

    if (!backendRes.ok) {
      throw new Error(
        `Backend feedback request failed (${backendRes.status}): ${JSON.stringify(backendPayload)}`,
      );
    }

    return NextResponse.json({ success: true, data: backendPayload }, { status: 200 });
  } catch (error) {
    console.error("Feedback API route error:", error);
    return NextResponse.json(
      { success: false, error: "Failed to submit feedback" },
      { status: 500 },
    );
  }
}
