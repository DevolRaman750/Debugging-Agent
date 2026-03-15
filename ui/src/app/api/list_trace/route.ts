import { NextResponse } from "next/server";
import { Trace, TraceResponse } from "@/models/trace";
import { getAuthTokenAndHeaders, createFetchHeaders } from "@/lib/clerk-auth";

const FETCH_TIMEOUT = 60_000; // 60 seconds

/**
 * Map a Python backend trace object (trace_id) to the frontend
 * Trace interface (id). Fields absent from the backend response
 * receive sensible defaults so the UI doesn't break.
 */
function normalizePythonTrace(raw: Record<string, unknown>): Trace {
  return {
    id: (raw.trace_id ?? raw.id ?? "") as string,
    service_name: (raw.service_name as string) ?? undefined,
    service_environment: (raw.service_environment as string) ?? undefined,
    duration: (raw.duration as number) ?? 0,
    start_time: (raw.start_time as number) ?? 0,
    end_time: (raw.end_time as number) ?? 0,
    percentile: (raw.percentile as string) ?? "P50",
    spans: (raw.spans as Trace["spans"]) ?? [],
    telemetry_sdk_language:
      (raw.telemetry_sdk_language as string[]) ?? [],
    // Optional log-count fields — pass through if present
    num_debug_logs: raw.num_debug_logs as number | undefined,
    num_info_logs: raw.num_info_logs as number | undefined,
    num_warning_logs: raw.num_warning_logs as number | undefined,
    num_error_logs: raw.num_error_logs as number | undefined,
    num_critical_logs: raw.num_critical_logs as number | undefined,
  };
}

// ── GET /api/list_trace ──────────────────────────────────────
// Proxies to REST_API_ENDPOINT/v1/explore/list-traces
// See: TRACEROOT_UI_REFERENCE.md — FLOW 1 (Trace Fetching)
export async function GET(
  request: Request,
): Promise<NextResponse<TraceResponse>> {
  try {
    // ── 1. Auth ──────────────────────────────────────────────
    const authResult = await getAuthTokenAndHeaders(request);

    if (!authResult) {
      return NextResponse.json(
        {
          success: false,
          data: [],
          error: "Authentication required. Please sign in.",
        },
        { status: 401 },
      );
    }

    // ── 2. Parse query params ────────────────────────────────
    const { searchParams } = new URL(request.url);
    const startTime = searchParams.get("startTime");
    const endTime = searchParams.get("endTime");
    const categories = searchParams.getAll("categories");
    const values = searchParams.getAll("values");
    const operations = searchParams.getAll("operations");
    const traceProvider = searchParams.get("trace_provider");
    const traceRegion = searchParams.get("trace_region");
    const logProvider = searchParams.get("log_provider");
    const logRegion = searchParams.get("log_region");
    const traceId = searchParams.get("trace_id");
    const paginationToken = searchParams.get("pagination_token");

    // ── 3. Validate env ─────────────────────────────────────
    const restApiEndpoint = process.env.REST_API_ENDPOINT;

    if (!restApiEndpoint) {
      return NextResponse.json(
        {
          success: false,
          data: [],
          error: "REST_API_ENDPOINT environment variable is not set",
        },
        { status: 500 },
      );
    }

    // ── 4. Default time window (last 3 hours) ───────────────
    const now = new Date();
    const threeHoursAgo = new Date(now.getTime() - 3 * 60 * 60 * 1000);
    const startTimeValue = startTime || threeHoursAgo.toISOString();
    const endTimeValue = endTime || now.toISOString();

    // ── 5. Build upstream URL ────────────────────────────────
    const apiUrl = new URL(
      `${restApiEndpoint}/v1/explore/list-traces`,
    );
    apiUrl.searchParams.set("start_time", startTimeValue);
    apiUrl.searchParams.set("end_time", endTimeValue);

    // Multi-value filter params
    for (const c of categories) apiUrl.searchParams.append("categories", c);
    for (const v of values) apiUrl.searchParams.append("values", v);
    for (const o of operations) apiUrl.searchParams.append("operations", o);

    // Provider (required — default to "aws")
    apiUrl.searchParams.set(
      "trace_provider",
      traceProvider || "aws",
    );
    apiUrl.searchParams.set(
      "log_provider",
      logProvider || "aws",
    );

    // Regions (optional — only relevant for AWS / Tencent)
    if (traceRegion) apiUrl.searchParams.set("trace_region", traceRegion);
    if (logRegion) apiUrl.searchParams.set("log_region", logRegion);

    // Direct trace lookup
    if (traceId) apiUrl.searchParams.set("trace_id", traceId);

    // Pagination
    if (paginationToken) {
      apiUrl.searchParams.set("pagination_token", paginationToken);
    }

    // ── 6. Fetch from Python backend ────────────────────────
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

    const response = await fetch(apiUrl.toString(), {
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

    // ── 7. Normalise response ────────────────────────────────
    const pythonResponse = await response.json();

    // Python model uses trace_id; frontend Trace uses id.
    const traces: Trace[] = (
      pythonResponse.traces as Record<string, unknown>[]
    ).map(normalizePythonTrace);

    return NextResponse.json({
      success: true,
      data: traces,
      next_pagination_token: pythonResponse.next_pagination_token,
      has_more: pythonResponse.has_more,
    });
  } catch (error: unknown) {
    console.error("[list_trace] Error fetching traces:", error);
    return NextResponse.json(
      {
        success: false,
        data: [],
        error:
          error instanceof Error ? error.message : "Failed to fetch trace data",
      },
      { status: 500 },
    );
  }
}
