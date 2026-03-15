// ══════════════════════════════════════════════════════════════
// MongoDB Connection Utility
// ══════════════════════════════════════════════════════════════
// Used only by direct MongoDB routes (provider-config, tokens).
// Skipped entirely when NEXT_PUBLIC_ENABLE_DIRECT_MONGO=false.

import mongoose from "mongoose";

const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
  console.warn(
    "[TraceRoot] MONGODB_URI not set — direct MongoDB routes disabled. " +
    "All data flows through the Python backend."
  );
}

interface MongooseCache {
  conn: typeof mongoose | null;
  promise: Promise<typeof mongoose> | null;
}

// Use global cache to prevent reconnecting on every API call in dev mode
const globalWithMongoose = global as typeof global & { mongoose?: MongooseCache };
const cached: MongooseCache = globalWithMongoose.mongoose ?? { conn: null, promise: null };
globalWithMongoose.mongoose = cached;

/**
 * Check whether MongoDB is configured (MONGODB_URI is set).
 */
export function isMongoDBAvailable(): boolean {
  return !!MONGODB_URI;
}

/**
 * Connect to MongoDB. Returns the mongoose instance.
 * Throws if MONGODB_URI is not configured.
 */
export async function connectToMongoDB(): Promise<typeof mongoose> {
  if (!MONGODB_URI) {
    throw new Error(
      "MONGODB_URI is not configured. Set it in .env.local or disable direct MongoDB routes."
    );
  }

  if (cached.conn) return cached.conn;

  if (!cached.promise) {
    cached.promise = mongoose.connect(MONGODB_URI, {
      bufferCommands: false,
    });
  }

  cached.conn = await cached.promise;
  return cached.conn;
}

/** Alias for backward compatibility with existing API routes. */
export const connectToDatabase = connectToMongoDB;

export default connectToMongoDB;
