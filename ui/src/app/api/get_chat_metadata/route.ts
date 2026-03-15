import { NextResponse } from "next/server";
import { createBackendAuthHeaders } from "@/lib/server-auth-headers";

interface ChatMetadataResponse {
  chat_id: string;
  chat_title?: string;
  trace_id?: string;
  timestamp?: number | string;
  error?: string;
  [key: string]: unknown;
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const chatId = url.searchParams.get("chat_id");

    if (!chatId) {
      return NextResponse.json(
        { success: false, error: "chat_id query parameter is required" },
        { status: 400 },
      );
    }

    const restApiEndpoint = process.env.REST_API_ENDPOINT;
    if (!restApiEndpoint) {
      return NextResponse.json(
        { success: false, error: "REST_API_ENDPOINT is not configured" },
        { status: 500 },
      );
    }

    const headers = await createBackendAuthHeaders(request);
    const apiUrl = `${restApiEndpoint}/v1/explore/get-chat-metadata?chat_id=${encodeURIComponent(chatId)}`;

    const upstreamResponse = await fetch(apiUrl, {
      method: "GET",
      headers,
    });

    const rawData: ChatMetadataResponse = await upstreamResponse.json();

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          success: false,
          error: rawData.error || `Upstream returned ${upstreamResponse.status}`,
        },
        { status: upstreamResponse.status },
      );
    }

    const normalizedTimestamp =
      typeof rawData.timestamp === "string"
        ? new Date(rawData.timestamp).getTime()
        : rawData.timestamp;

    return NextResponse.json({
      ...rawData,
      timestamp: normalizedTimestamp,
    });
  } catch (error) {
    console.error("GET /api/get_chat_metadata failed:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch chat metadata" },
      { status: 500 },
    );
  }
}
