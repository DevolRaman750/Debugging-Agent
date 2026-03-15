/**
 * @jest-environment node
 */

/**
 * Unit tests for GET /api/get_trace_log
 *
 * Tests:
 * - Returns 401 when unauthenticated
 * - Returns 400 when traceId is missing
 * - Returns 500 when REST_API_ENDPOINT is missing
 * - Builds correct upstream URL
 * - Transforms Python response → TraceLog format
 * - Handles upstream errors
 */

// ── Mocks ────────────────────────────────────────────────────

const mockGetAuthTokenAndHeaders = jest.fn();
const mockCreateFetchHeaders = jest.fn();
jest.mock("@/lib/clerk-auth", () => ({
  getAuthTokenAndHeaders: (...args: unknown[]) =>
    mockGetAuthTokenAndHeaders(...args),
  createFetchHeaders: (...args: unknown[]) =>
    mockCreateFetchHeaders(...args),
}));

const mockFetch = jest.fn();
global.fetch = mockFetch;

import { GET } from "@/app/api/get_trace_log/route";

// ── Helpers ──────────────────────────────────────────────────

function makeRequest(params: Record<string, string> = {}): Request {
  const url = new URL("http://localhost:3000/api/get_trace_log");
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, value);
  }
  return new Request(url.toString());
}

const MOCK_AUTH_RESULT = {
  user: {
    userId: "user_123",
    email: "test@example.com",
    fullName: "Test User",
    imageUrl: null,
    isAuthenticated: true,
  },
  token: "test-jwt-token",
};

const MOCK_PYTHON_LOG_RESPONSE = {
  logs: {
    logs: [
      {
        span_id_1: [
          {
            time: 1700000000,
            level: "ERROR",
            message: "Something went wrong",
            file_name: "main.py",
            function_name: "handle_request",
            line_number: 42,
            span_id: "span_id_1",
          },
        ],
      },
      {
        span_id_2: [
          {
            time: 1700000001,
            level: "INFO",
            message: "Request processed",
            file_name: "handler.py",
            function_name: "process",
            line_number: 15,
            span_id: "span_id_2",
          },
        ],
      },
    ],
  },
};

// ── Tests ────────────────────────────────────────────────────

describe("GET /api/get_trace_log", () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...ORIGINAL_ENV, REST_API_ENDPOINT: "http://backend:8000" };
    mockCreateFetchHeaders.mockReturnValue({
      "Content-Type": "application/json",
    });
  });

  afterEach(() => {
    process.env = ORIGINAL_ENV;
  });

  // ── Auth ──────────────────────────────────────────────────

  it("returns 401 when user is not authenticated", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(null);

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.success).toBe(false);
    expect(body.error).toContain("Authentication required");
  });

  // ── Validation ────────────────────────────────────────────

  it("returns 400 when traceId is missing", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);

    const response = await GET(makeRequest({}));
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body.success).toBe(false);
    expect(body.error).toContain("Trace ID is required");
  });

  // ── Env validation ────────────────────────────────────────

  it("returns 500 when REST_API_ENDPOINT is not set", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    delete process.env.REST_API_ENDPOINT;

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("REST_API_ENDPOINT");
  });

  // ── Successful fetch ──────────────────────────────────────

  it("fetches logs and transforms to TraceLog format", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => MOCK_PYTHON_LOG_RESPONSE,
    });

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    // The response should be keyed by traceId
    expect(body.data).toHaveProperty("trace123");
    // The value should be the array of SpanLog objects
    expect(body.data["trace123"]).toEqual(MOCK_PYTHON_LOG_RESPONSE.logs.logs);
  });

  it("returns null data when logs are empty", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ logs: {} }),
    });

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toBeNull();
  });

  // ── URL building ──────────────────────────────────────────

  it("builds correct upstream URL with all optional params", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ logs: { logs: [] } }),
    });

    await GET(
      makeRequest({
        traceId: "trace123",
        start_time: "2024-01-01T00:00:00Z",
        end_time: "2024-01-01T03:00:00Z",
        log_group_name: "/ecs/my-service",
        trace_provider: "aws",
        trace_region: "us-east-1",
        log_provider: "aws",
        log_region: "us-west-2",
      }),
    );

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const calledUrl = new URL(mockFetch.mock.calls[0][0]);

    expect(calledUrl.pathname).toBe("/v1/explore/get-logs-by-trace-id");
    expect(calledUrl.searchParams.get("trace_id")).toBe("trace123");
    expect(calledUrl.searchParams.get("start_time")).toBe(
      "2024-01-01T00:00:00Z",
    );
    expect(calledUrl.searchParams.get("end_time")).toBe(
      "2024-01-01T03:00:00Z",
    );
    expect(calledUrl.searchParams.get("log_group_name")).toBe("/ecs/my-service");
    expect(calledUrl.searchParams.get("trace_provider")).toBe("aws");
    expect(calledUrl.searchParams.get("trace_region")).toBe("us-east-1");
    expect(calledUrl.searchParams.get("log_provider")).toBe("aws");
    expect(calledUrl.searchParams.get("log_region")).toBe("us-west-2");
  });

  it("omits optional params when not provided", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ logs: { logs: [] } }),
    });

    await GET(makeRequest({ traceId: "trace123" }));

    const calledUrl = new URL(mockFetch.mock.calls[0][0]);
    expect(calledUrl.searchParams.get("trace_id")).toBe("trace123");
    expect(calledUrl.searchParams.has("start_time")).toBe(false);
    expect(calledUrl.searchParams.has("log_group_name")).toBe(false);
    expect(calledUrl.searchParams.has("trace_provider")).toBe(false);
  });

  // ── Error handling ────────────────────────────────────────

  it("returns 500 when upstream returns non-ok response", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
      text: async () => "backend crashed",
    });

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("500");
  });

  it("returns 500 when fetch throws", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockRejectedValue(new Error("Network failure"));

    const response = await GET(makeRequest({ traceId: "trace123" }));
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("Network failure");
  });
});
