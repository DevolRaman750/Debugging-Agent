/**
 * @jest-environment node
 */

/**
 * Unit tests for GET /api/list_trace
 *
 * Tests:
 * - Returns 401 when unauthenticated (Clerk active)
 * - Returns 500 when REST_API_ENDPOINT is missing
 * - Builds correct upstream URL with query params
 * - Normalizes Python trace_id → frontend id
 * - Returns default time window when none provided
 * - Passes pagination_token through
 * - Handles upstream errors gracefully
 */

// ── Mocks must be set up before imports ──────────────────────

// Mock clerk-auth
const mockGetAuthTokenAndHeaders = jest.fn();
const mockCreateFetchHeaders = jest.fn();
jest.mock("@/lib/clerk-auth", () => ({
  getAuthTokenAndHeaders: (...args: unknown[]) =>
    mockGetAuthTokenAndHeaders(...args),
  createFetchHeaders: (...args: unknown[]) =>
    mockCreateFetchHeaders(...args),
}));

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Import after mocks
import { GET } from "@/app/api/list_trace/route";

// ── Helpers ──────────────────────────────────────────────────

function makeRequest(params: Record<string, string | string[]> = {}): Request {
  const url = new URL("http://localhost:3000/api/list_trace");
  for (const [key, value] of Object.entries(params)) {
    if (Array.isArray(value)) {
      value.forEach((v) => url.searchParams.append(key, v));
    } else {
      url.searchParams.set(key, value);
    }
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

const MOCK_PYTHON_RESPONSE = {
  traces: [
    {
      trace_id: "abc123",
      service_name: "my-service",
      duration: 1.5,
      start_time: 1700000000,
      end_time: 1700000001,
      percentile: "P90",
      spans: [],
      telemetry_sdk_language: ["python"],
      num_error_logs: 2,
    },
    {
      trace_id: "def456",
      service_name: "other-service",
      duration: 0.3,
      start_time: 1700000010,
      end_time: 1700000011,
      spans: [],
    },
  ],
  next_pagination_token: "page2_token",
  has_more: true,
};

// ── Tests ────────────────────────────────────────────────────

describe("GET /api/list_trace", () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...ORIGINAL_ENV, REST_API_ENDPOINT: "http://backend:8000" };
    mockCreateFetchHeaders.mockReturnValue({
      "Content-Type": "application/json",
      Authorization: "Bearer test-jwt-token",
    });
  });

  afterEach(() => {
    process.env = ORIGINAL_ENV;
  });

  // ── Auth ──────────────────────────────────────────────────

  it("returns 401 when user is not authenticated", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(null);

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(401);
    expect(body.success).toBe(false);
    expect(body.error).toContain("Authentication required");
  });

  // ── Env validation ────────────────────────────────────────

  it("returns 500 when REST_API_ENDPOINT is not set", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    delete process.env.REST_API_ENDPOINT;

    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("REST_API_ENDPOINT");
  });

  // ── Successful fetch ──────────────────────────────────────

  it("fetches traces and normalizes Python trace_id → id", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => MOCK_PYTHON_RESPONSE,
    });

    const response = await GET(
      makeRequest({
        startTime: "2024-01-01T00:00:00Z",
        endTime: "2024-01-01T03:00:00Z",
      }),
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data).toHaveLength(2);

    // Verify trace_id → id mapping
    expect(body.data[0].id).toBe("abc123");
    expect(body.data[0].service_name).toBe("my-service");
    expect(body.data[0].percentile).toBe("P90");
    expect(body.data[0].num_error_logs).toBe(2);

    // Second trace should get defaults for missing fields
    expect(body.data[1].id).toBe("def456");
    expect(body.data[1].percentile).toBe("P50"); // default
    expect(body.data[1].telemetry_sdk_language).toEqual([]); // default
  });

  it("passes pagination info through", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => MOCK_PYTHON_RESPONSE,
    });

    const response = await GET(
      makeRequest({
        startTime: "2024-01-01T00:00:00Z",
        endTime: "2024-01-01T03:00:00Z",
      }),
    );
    const body = await response.json();

    expect(body.next_pagination_token).toBe("page2_token");
    expect(body.has_more).toBe(true);
  });

  // ── URL building ──────────────────────────────────────────

  it("builds correct upstream URL with all query params", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ traces: [] }),
    });

    await GET(
      makeRequest({
        startTime: "2024-01-01T00:00:00Z",
        endTime: "2024-01-01T03:00:00Z",
        trace_provider: "jaeger",
        log_provider: "jaeger",
        trace_region: "us-east-1",
        categories: ["service_name", "span_name"],
        values: ["my-service", "handler"],
        operations: ["=", "contains"],
        pagination_token: "tok123",
        trace_id: "specific-trace",
      }),
    );

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const calledUrl = new URL(mockFetch.mock.calls[0][0]);

    expect(calledUrl.pathname).toBe("/v1/explore/list-traces");
    expect(calledUrl.searchParams.get("start_time")).toBe(
      "2024-01-01T00:00:00Z",
    );
    expect(calledUrl.searchParams.get("end_time")).toBe(
      "2024-01-01T03:00:00Z",
    );
    expect(calledUrl.searchParams.get("trace_provider")).toBe("jaeger");
    expect(calledUrl.searchParams.get("log_provider")).toBe("jaeger");
    expect(calledUrl.searchParams.get("trace_region")).toBe("us-east-1");
    expect(calledUrl.searchParams.getAll("categories")).toEqual([
      "service_name",
      "span_name",
    ]);
    expect(calledUrl.searchParams.getAll("values")).toEqual([
      "my-service",
      "handler",
    ]);
    expect(calledUrl.searchParams.getAll("operations")).toEqual([
      "=",
      "contains",
    ]);
    expect(calledUrl.searchParams.get("pagination_token")).toBe("tok123");
    expect(calledUrl.searchParams.get("trace_id")).toBe("specific-trace");
  });

  it("defaults trace_provider and log_provider to 'aws' when not provided", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ traces: [] }),
    });

    await GET(
      makeRequest({
        startTime: "2024-01-01T00:00:00Z",
        endTime: "2024-01-01T03:00:00Z",
      }),
    );

    const calledUrl = new URL(mockFetch.mock.calls[0][0]);
    expect(calledUrl.searchParams.get("trace_provider")).toBe("aws");
    expect(calledUrl.searchParams.get("log_provider")).toBe("aws");
  });

  it("uses default 3-hour time window when no times provided", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ traces: [] }),
    });

    const before = Date.now();
    await GET(makeRequest());
    const after = Date.now();

    const calledUrl = new URL(mockFetch.mock.calls[0][0]);
    const startTime = new Date(
      calledUrl.searchParams.get("start_time")!,
    ).getTime();
    const endTime = new Date(
      calledUrl.searchParams.get("end_time")!,
    ).getTime();

    // end_time should be roughly "now"
    expect(endTime).toBeGreaterThanOrEqual(before - 1000);
    expect(endTime).toBeLessThanOrEqual(after + 1000);

    // start_time should be ~3 hours before end_time
    const threeHoursMs = 3 * 60 * 60 * 1000;
    expect(endTime - startTime).toBeCloseTo(threeHoursMs, -3); // within 1s
  });

  // ── Auth headers forwarding ───────────────────────────────

  it("forwards auth headers to the Python backend", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockCreateFetchHeaders.mockReturnValue({
      "Content-Type": "application/json",
      Authorization: "Bearer test-jwt-token",
      "X-User-Email": "test@example.com",
    });
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({ traces: [] }),
    });

    await GET(makeRequest({ startTime: "2024-01-01T00:00:00Z" }));

    expect(mockCreateFetchHeaders).toHaveBeenCalledWith(MOCK_AUTH_RESULT);
    const fetchOptions = mockFetch.mock.calls[0][1];
    expect(fetchOptions.headers).toEqual(
      expect.objectContaining({
        Authorization: "Bearer test-jwt-token",
        "X-User-Email": "test@example.com",
      }),
    );
  });

  // ── Error handling ────────────────────────────────────────

  it("returns 500 when upstream returns non-ok response", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockResolvedValue({
      ok: false,
      status: 502,
      statusText: "Bad Gateway",
      text: async () => "upstream error",
    });

    const response = await GET(
      makeRequest({ startTime: "2024-01-01T00:00:00Z" }),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("502");
  });

  it("returns 500 when fetch throws (network error)", async () => {
    mockGetAuthTokenAndHeaders.mockResolvedValue(MOCK_AUTH_RESULT);
    mockFetch.mockRejectedValue(new Error("ECONNREFUSED"));

    const response = await GET(
      makeRequest({ startTime: "2024-01-01T00:00:00Z" }),
    );
    const body = await response.json();

    expect(response.status).toBe(500);
    expect(body.success).toBe(false);
    expect(body.error).toContain("ECONNREFUSED");
  });
});
