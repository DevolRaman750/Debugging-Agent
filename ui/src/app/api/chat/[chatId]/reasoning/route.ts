import { NextResponse } from "next/server";
import mongoose, { Model, Schema } from "mongoose";
import dbConnect from "@/lib/mongodb";
import { createBackendAuthHeaders } from "@/lib/server-auth-headers";

interface ReasoningStreamDocument extends mongoose.Document {
  chat_id: string;
  chunk_id: number;
  content: string;
  status: "in_progress" | "complete";
  timestamp: Date;
}

const reasoningStreamSchema = new Schema<ReasoningStreamDocument>(
  {
    chat_id: { type: String, required: true, index: true },
    chunk_id: { type: Number, required: true },
    content: { type: String, required: true },
    status: {
      type: String,
      enum: ["in_progress", "complete"],
      required: true,
      default: "in_progress",
    },
    timestamp: { type: Date, required: true, default: Date.now },
  },
  {
    collection: process.env.DB_REASONING_COLLECTION || "reasoning_streams",
    versionKey: false,
  },
);

const ReasoningStream: Model<ReasoningStreamDocument> =
  (mongoose.models.ReasoningStream as Model<ReasoningStreamDocument>) ||
  mongoose.model<ReasoningStreamDocument>(
    "ReasoningStream",
    reasoningStreamSchema,
  );

export async function GET(
  request: Request,
  { params }: { params: Promise<{ chatId: string }> },
) {
  try {
    const { chatId } = await params;

    if (!chatId) {
      return NextResponse.json(
        { success: false, error: "chatId is required" },
        { status: 400 },
      );
    }

    const url = new URL(request.url);
    const after = url.searchParams.get("after");

    // If direct MongoDB is disabled, proxy reasoning fetch to Python backend.
    if (!process.env.MONGODB_URI) {
      const restApiEndpoint = process.env.REST_API_ENDPOINT;
      if (!restApiEndpoint) {
        return NextResponse.json({ success: true, data: [] });
      }

      const backendUrl = new URL(`${restApiEndpoint}/v1/explore/get-chat-reasoning`);
      backendUrl.searchParams.set("chat_id", chatId);
      if (after) {
        backendUrl.searchParams.set("after", after);
      }

      const headers = await createBackendAuthHeaders(request);
      const upstreamRes = await fetch(backendUrl.toString(), {
        method: "GET",
        headers,
      });

      const upstreamData = await upstreamRes.json().catch(() => ({}));
      if (!upstreamRes.ok) {
        return NextResponse.json({ success: true, data: [] });
      }

      const list: unknown[] = Array.isArray(upstreamData?.reasoning)
        ? upstreamData.reasoning
        : Array.isArray(upstreamData?.data)
          ? upstreamData.data
          : [];

      return NextResponse.json({ success: true, data: list });
    }

    await dbConnect();

    const query: { chat_id: string; timestamp?: { $gt: Date } } = {
      chat_id: chatId,
    };

    if (after) {
      const parsedAfter = Number(after);
      if (!Number.isNaN(parsedAfter)) {
        query.timestamp = { $gt: new Date(parsedAfter) };
      }
    }

    const results = await ReasoningStream.find(query)
      .sort({ timestamp: 1 })
      .lean();

    return NextResponse.json({ success: true, data: results });
  } catch (error) {
    console.error("GET /api/chat/[chatId]/reasoning failed:", error);
    return NextResponse.json(
      { success: false, error: "Failed to fetch reasoning stream" },
      { status: 500 },
    );
  }
}
