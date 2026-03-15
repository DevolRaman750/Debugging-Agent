import { NextResponse } from "next/server";
import { createBackendAuthHeaders } from "@/lib/server-auth-headers";

interface ChatMetadataItem {
  chat_id: string;
  chat_title?: string;
  trace_id?: string;
  timestamp: number | string;
  [key: string]: unknown;
}

interface UpstreamMetadataHistoryResponse {
  success?: boolean;
  data?: {
    history?: ChatMetadataItem[];
  };
  hasMore?: boolean;
  error?: string;
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const traceId = url.searchParams.get("trace_id");
    const skip = url.searchParams.get("skip") ?? "0";
    const limit = url.searchParams.get("limit") ?? "50";

    if (!traceId) {
      return NextResponse.json(
        { success: false, error: "trace_id query parameter is required" },
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
    const backendUrl = new URL(
      `${restApiEndpoint}/v1/explore/get-chat-metadata-history`,
    );
    backendUrl.searchParams.set("trace_id", traceId);
    backendUrl.searchParams.set("skip", skip);
    backendUrl.searchParams.set("limit", limit);

    const upstreamResponse = await fetch(backendUrl.toString(), {
      method: "GET",
      headers,
    });

    const upstreamData: UpstreamMetadataHistoryResponse =
      await upstreamResponse.json();

    if (!upstreamResponse.ok) {
      return NextResponse.json(
        {
          success: false,
          error:
            upstreamData.error || `Upstream returned ${upstreamResponse.status}`,
        },
        { status: upstreamResponse.status },
      );
    }

    const history = (upstreamData.data?.history ?? []).map((item) => ({
      ...item,
      timestamp:
        typeof item.timestamp === "string"
          ? new Date(item.timestamp).getTime()
          : item.timestamp,
    }));

    return NextResponse.json({
      success: upstreamData.success ?? true,
      data: { history },
      hasMore: Boolean(upstreamData.hasMore),
    });
  } catch (error) {
    console.error("GET /api/get_chat_metadata_history failed:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch chat metadata history" },
      { status: 500 },
    );
  }
}
