import { NextResponse } from "next/server";
import { TraceLog } from "@/models/log";
import {
  getAuthTokenAndHeaders,
  createFetchHeaders,
} from "@/lib/clerk-auth";

const FETCH_TIMEOUT = 90_000; // 90 seconds

export interface LogResponse {
  success: boolean;
  data: TraceLog | null;
  error?: string;
}

// ── GET /api/get_trace_log ───────────────────────────────────
// Proxies to REST_API_ENDPOINT/v1/explore/get-logs-by-trace-id
// See: ROOTIX_UI_REFERENCE.md — FLOW 2 (Log Fetching)
export async function GET(
  request: Request,
): Promise<NextResponse<LogResponse>> {
  try {
    // ── 1. Auth ──────────────────────────────────────────────
    const authResult = await getAuthTokenAndHeaders(request);

    if (!authResult) {
      return NextResponse.json(
        {
          success: false,
          data: null,
          error: "Authentication required. Please sign in.",
        },
        { status: 401 },
      );
    }

    // ── 2. Parse query params ────────────────────────────────
    const { searchParams } = new URL(request.url);
    const traceId = searchParams.get("traceId");
    const startTime = searchParams.get("start_time");
    const endTime = searchParams.get("end_time");
    const logGroupName = searchParams.get("log_group_name");
    const traceProvider = searchParams.get("trace_provider");
    const traceRegion = searchParams.get("trace_region");
    const logProvider = searchParams.get("log_provider");
    const logRegion = searchParams.get("log_region");

    if (!traceId) {
      return NextResponse.json(
        { success: false, data: null, error: "Trace ID is required" },
        { status: 400 },
      );
    }

    // ── 3. Validate env ─────────────────────────────────────
    const restApiEndpoint = process.env.REST_API_ENDPOINT;

    if (!restApiEndpoint) {
      return NextResponse.json(
        {
          success: false,
          data: null,
          error: "REST_API_ENDPOINT environment variable is not set",
        },
        { status: 500 },
      );
    }

    // ── 4. Build upstream URL ────────────────────────────────
    const url = new URL(
      `${restApiEndpoint}/v1/explore/get-logs-by-trace-id`,
    );
    url.searchParams.set("trace_id", traceId);

    // Timestamps are optional — backend can search without them
    if (startTime) url.searchParams.set("start_time", startTime);
    if (endTime) url.searchParams.set("end_time", endTime);
    if (logGroupName) url.searchParams.set("log_group_name", logGroupName);

    // Provider params
    if (traceProvider) url.searchParams.set("trace_provider", traceProvider);
    if (traceRegion) url.searchParams.set("trace_region", traceRegion);
    if (logProvider) url.searchParams.set("log_provider", logProvider);
    if (logRegion) url.searchParams.set("log_region", logRegion);

    // ── 5. Fetch from Python backend ────────────────────────
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

    const response = await fetch(url.toString(), {
      method: "GET",
      headers: createFetchHeaders(authResult),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `REST API ${response.status} ${response.statusText}: ${body}`,
      );
    }

    // ── 6. Transform response ────────────────────────────────
    // Python returns { logs: { logs: LogEntry[] } }
    // Frontend expects TraceLog = { [traceId]: SpanLog[] }
    const apiResponse = await response.json();

    let logData: TraceLog | null = null;

    if (apiResponse.logs && apiResponse.logs.logs) {
      logData = { [traceId]: apiResponse.logs.logs };
    }

    return NextResponse.json({
      success: true,
      data: logData,
    });
  } catch (error: unknown) {
    console.error("[get_trace_log] Error fetching log data:", error);
    return NextResponse.json(
      {
        success: false,
        data: null,
        error:
          error instanceof Error ? error.message : "Failed to fetch log data",
      },
      { status: 500 },
    );
  }
}
