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

import { GET } from "@/app/api/get_chat_history/route";

function makeRequest(chatId?: string): Request {
  const url = new URL("http://localhost:3000/api/get_chat_history");
  if (chatId) {
    url.searchParams.set("chat_id", chatId);
  }
  return new Request(url.toString());
}

describe("GET /api/get_chat_history", () => {
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

  it("proxies get-chat-history and normalizes timestamp strings", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        history: [
          {
            time: "2026-03-14T01:00:00.000Z",
            message: "hello",
            message_type: "assistant",
            chat_id: "chat-1",
            reference: [],
          },
        ],
      }),
    });

    const response = await GET(makeRequest("chat-1"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(Array.isArray(body.history)).toBe(true);
    expect(typeof body.history[0].time).toBe("number");

    const [calledUrl] = mockFetch.mock.calls[0];
    expect(calledUrl).toContain("/v1/explore/get-chat-history?chat_id=chat-1");
  });

  it("returns empty history fallback when endpoint is missing", async () => {
    delete process.env.REST_API_ENDPOINT;

    const response = await GET(makeRequest("chat-1"));
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body).toEqual({ history: [] });
  });

  it("returns 400 with empty history when chat_id is missing", async () => {
    const response = await GET(makeRequest());
    const body = await response.json();

    expect(response.status).toBe(400);
    expect(body).toEqual({ history: [] });
  });
});
