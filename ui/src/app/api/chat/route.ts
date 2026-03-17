import { NextResponse } from "next/server";
import {
  ActionType,
  ActionStatus,
  ChatRequest,
  ChatResponse,
  ChatbotResponse,
  IntelligenceMetadata,
  MessageType,
} from "@/models/chat";
import { createBackendAuthHeaders } from "@/lib/server-auth-headers";

function toEpochMs(value: unknown): number {
  if (typeof value === "number") {
    return value;
  }

  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }

  return Date.now();
}

function toIsoString(value: unknown): string {
  if (typeof value === "number") {
    const ms = value > 1e12 ? value : value * 1000;
    return new Date(ms).toISOString();
  }

  if (typeof value === "string") {
    const parsed = Date.parse(value);
    if (!Number.isNaN(parsed)) {
      return new Date(parsed).toISOString();
    }
  }

  return new Date().toISOString();
}

export async function POST(
  request: Request,
): Promise<NextResponse<ChatResponse>> {
  try {
    const body: ChatRequest = await request.json();
    const {
      time,
      message,
      message_type,
      trace_id,
      span_ids,
      start_time,
      end_time,
      model,
      mode,
      chat_id,
      trace_provider,
      log_provider,
      trace_region,
      log_region,
      provider,
      providers,
    } = body;
    const restApiEndpoint = process.env.REST_API_ENDPOINT;

    const resolvedTraceProvider =
      providers?.trace_provider ?? trace_provider ?? "aws";
    const resolvedLogProvider = providers?.log_provider ?? log_provider ?? "aws";
    const resolvedTraceRegion = providers?.trace_region ?? trace_region;
    const resolvedLogRegion = providers?.log_region ?? log_region;
    const resolvedProvider = providers?.provider ?? provider ?? "openai";

    if (restApiEndpoint) {
      // Call the REST API endpoint
      try {
        const apiUrl = `${restApiEndpoint}/v1/explore/post-chat`;

        // Create the request body matching the ChatRequest structure from the REST API
        const apiRequestBody = {
          time: toIsoString(time),
          message,
          message_type: message_type ?? "user",
          trace_id,
          span_ids: span_ids ?? [],
          start_time: toIsoString(start_time),
          end_time: toIsoString(end_time),
          model: model ?? "auto",
          mode: mode ?? "agent",
          chat_id,
          trace_provider: resolvedTraceProvider,
          log_provider: resolvedLogProvider,
          service_name: null,
          trace_region: resolvedTraceRegion,
          log_region: resolvedLogRegion,
          provider: resolvedProvider,
        };

        // Get auth headers (automatically uses Clerk's auth() and currentUser())
        const headers = await createBackendAuthHeaders();
        const apiResponse = await fetch(apiUrl, {
          method: "POST",
          headers,
          body: JSON.stringify(apiRequestBody),
        });

        let finalResponse = apiResponse;

        // Compatibility retry for backends with stricter/different request schemas.
        if (apiResponse.status === 422) {
          const compatibilityBody = {
            time: toEpochMs(time),
            message,
            trace_id,
            span_ids: span_ids ?? [],
            start_time: toEpochMs(start_time),
            end_time: toEpochMs(end_time),
            model: model === "llama-3.3-70b-versatile" ? "auto" : model ?? "auto",
            mode:
              typeof mode === "string"
                ? mode.toUpperCase() === "CHAT"
                  ? "CHAT"
                  : "AGENT"
                : "AGENT",
            chat_id,
            service_name: null,
          };

          finalResponse = await fetch(apiUrl, {
            method: "POST",
            headers,
            body: JSON.stringify(compatibilityBody),
          });
        }

        if (!finalResponse.ok) {
          const errorBody = await finalResponse.text();
          throw new Error(
            `REST API call failed with status: ${finalResponse.status}; body: ${errorBody}`,
          );
        }

        const apiData = await finalResponse.json();

        const responseMetadata = apiData?.metadata as
          | IntelligenceMetadata
          | undefined;

        // Transform the REST API response to match our ChatResponse format
        const chatbotResponse: ChatbotResponse = {
          time: toEpochMs(apiData?.time),
          message: apiData?.message || "",
          reference: Array.isArray(apiData?.reference) ? apiData.reference : [],
          message_type: "assistant" as MessageType,
          chat_id: apiData?.chat_id || chat_id,
          action_type: apiData.action_type,
          status: apiData.status,
          metadata: responseMetadata,
        };

        const response: ChatResponse = {
          success: true,
          data: chatbotResponse,
        };

        return NextResponse.json(response);
      } catch (apiError) {
        console.error("REST API call failed:", apiError);
        // Fall back to static response if REST API fails
        console.log("Falling back to static response due to API error");
      }
    }

    // Fallback to static response (original functionality)
    // This runs when either REST_API_ENDPOINT is not set or the API call fails
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Generate static response using the same logic from Agent.tsx
    const responseMessage = getStaticResponse(message, trace_id, span_ids || []);

    // Create response using Chat.ts models
    const chatbotResponse: ChatbotResponse = {
      time: new Date().getTime(),
      message: responseMessage,
      reference: [],
      message_type: "assistant" as MessageType,
      chat_id: chat_id,
      action_type: "agent_chat" as ActionType,
      status: "success" as ActionStatus,
      metadata: {
        confidence: "LOW",
        pattern_matched: null,
        fast_path: true,
        processing_time_ms: 0,
        top_cause: null,
        top_cause_score: null,
        causes_found: 0,
      },
    };

    const response: ChatResponse = {
      success: true,
      data: chatbotResponse,
    };

    return NextResponse.json(response);
  } catch (error) {
    console.error("Chat API Error:", error);

    const errorResponse: ChatResponse = {
      success: false,
      data: null,
      error: "Failed to process chat request",
    };

    return NextResponse.json(errorResponse, { status: 500 });
  }
}

function getStaticResponse(
  userInput: string,
  trace_id: string,
  span_ids: string[],
): string {
  const hasTrace = Boolean(trace_id);
  const hasSpans = span_ids.length > 0;

  const responses = {
    noContext: `I notice you haven't selected any trace or spans yet. To help you better, please select a trace and specific spans you'd like to analyze.`,
    traceOnly: `I can see you're looking at trace ${trace_id}. To provide more detailed analysis, please select specific spans within this trace.`,
    spansOnly: `I see you've selected ${span_ids.length} spans, but no trace is selected. For better context, please select a trace as well.`,
    fullContext: `I'm analyzing trace ${trace_id} with ${span_ids.length} selected spans. What specific aspect would you like to know about these spans?`,
  };

  if (!hasTrace && !hasSpans) {
    return responses.noContext;
  } else if (hasTrace && !hasSpans) {
    return responses.traceOnly;
  } else if (!hasTrace && hasSpans) {
    return responses.spansOnly;
  } else {
    const input = userInput.toLowerCase();
    if (input.includes("error") || input.includes("fail")) {
      return `${responses.fullContext}\n\nI can help you analyze any errors or failures in these spans. Would you like to see the error rates or specific error messages?`;
    } else if (
      input.includes("time") ||
      input.includes("duration") ||
      input.includes("slow")
    ) {
      return `${responses.fullContext}\n\nI can help you analyze the performance of these spans. Would you like to see the duration statistics or identify slow operations?`;
    } else if (input.includes("flow") || input.includes("sequence")) {
      return `${responses.fullContext}\n\nI can help you understand the flow of these spans. Would you like to see the sequence of operations or the dependencies between spans?`;
    } else {
      return `${responses.fullContext}\n\nI can help you analyze:\n- Performance metrics\n- Error patterns\n- Span relationships\n- Resource usage\nWhat would you like to know?`;
    }
  }
}
