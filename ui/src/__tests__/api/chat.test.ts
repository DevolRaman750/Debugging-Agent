/**
 * @jest-environment node
 */

const mockCreateBackendAuthHeaders = jest.fn();
jest.mock("@/lib/server-auth-headers", () => ({
  createBackendAuthHeaders: (...args: unknown[]) =>
    mockCreateBackendAuthHeaders(...args),
}));

const mockFetch = jest.fn();
global.fetch = mockFetch;

import { POST } from "@/app/api/chat/route";

function makeRequest(body: Record<string, unknown>): Request {
  return new Request("http://localhost:3000/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("POST /api/chat", () => {
  const ORIGINAL_ENV = process.env;

  beforeEach(() => {
    jest.clearAllMocks();
    process.env = { ...ORIGINAL_ENV, REST_API_ENDPOINT: "http://backend:8000" };
    mockCreateBackendAuthHeaders.mockResolvedValue({
      "Content-Type": "application/json",
      Authorization: "Bearer test-token",
    });
  });

  afterEach(() => {
    process.env = ORIGINAL_ENV;
  });

  it("proxies to post-chat and maps metadata in response", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        time: "2026-03-14T01:00:00.000Z",
        message: "Assistant answer",
        reference: [{ number: 1, span_id: "span-1" }],
        chat_id: "chat-123",
        action_type: "agent_chat",
        status: "success",
        metadata: {
          confidence: "HIGH",
          pattern_matched: "latency_spike",
          fast_path: false,
          processing_time_ms: 1200,
          top_cause: "database",
          top_cause_score: 0.92,
          causes_found: 2,
        },
      }),
    });

    const requestBody = {
      time: Date.now(),
      message: "why is this trace slow?",
      message_type: "user",
      trace_id: "trace-1",
      span_ids: ["span-1"],
      start_time: Date.now() - 1000,
      end_time: Date.now(),
      model: "gpt-4o",
      mode: "agent",
      chat_id: "chat-123",
      provider: "openai",
      providers: {
        trace_provider: "jaeger",
        log_provider: "jaeger",
        trace_region: "us-east-1",
        log_region: "us-east-1",
      },
    };

    const response = await POST(makeRequest(requestBody));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data.message).toBe("Assistant answer");
    expect(body.data.metadata.confidence).toBe("HIGH");

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0];
    expect(url).toBe("http://backend:8000/v1/explore/post-chat");
    expect(options.method).toBe("POST");

    const proxiedBody = JSON.parse(options.body);
    expect(proxiedBody.message_type).toBe("user");
    expect(proxiedBody.trace_provider).toBe("jaeger");
    expect(proxiedBody.log_provider).toBe("jaeger");
  });

  it("falls back to static response when REST_API_ENDPOINT is missing", async () => {
    delete process.env.REST_API_ENDPOINT;

    const response = await POST(
      makeRequest({
        message: "help me",
        message_type: "user",
        trace_id: "trace-1",
        span_ids: ["span-1"],
        model: "gpt-4o",
        mode: "agent",
        chat_id: "chat-fallback",
        provider: "openai",
      }),
    );

    const body = await response.json();
    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(typeof body.data.message).toBe("string");
    expect(body.data.metadata).toBeTruthy();
    expect(mockFetch).not.toHaveBeenCalled();
  });
});
