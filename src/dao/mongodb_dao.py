"""
MongoDB Persistence Layer for TraceRoot
=========================================
Production database using motor (async MongoDB driver).

Same interface as TraceRootSQLiteClient — both are interchangeable.

5 Collections:
  1. chat_records          — Every user/assistant message
  2. chat_metadata         — Chat session index
  3. reasoning_records     — LLM thinking chunks
  4. chat_routing          — Agent routing decisions
  5. intelligence_metrics  — Intel Layer results + validation + feedback

SETUP:
  pip install motor
  export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net/traceroot"
"""

import os
from datetime import datetime, timezone
from typing import Any

from src.dao.types import ChatMetadata, ChatMetadataHistory

# MongoDB URI from environment
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "traceroot")


class TraceRootMongoDBClient:

    def __init__(self, uri: str = MONGODB_URI, db_name: str = MONGODB_DB):
        # Lazy import — motor is only needed if using MongoDB
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            self.client = AsyncIOMotorClient(uri)
            self.db = self.client[db_name]
        except ImportError:
            raise ImportError(
                "motor is required for MongoDB support. "
                "Install it with: pip install motor"
            )

    async def _init_db(self):
        """Create indexes for MongoDB collections.

        Unlike SQLite, MongoDB auto-creates collections on first insert.
        We only need to set up indexes.
        """
        # Chat records indexes
        await self.db.chat_records.create_index("chat_id")
        await self.db.chat_records.create_index("timestamp")

        # Chat metadata indexes
        await self.db.chat_metadata.create_index("chat_id", unique=True)
        await self.db.chat_metadata.create_index("trace_id")

        # Reasoning indexes
        await self.db.reasoning_records.create_index("chat_id")
        await self.db.reasoning_records.create_index([("chat_id", 1), ("chunk_id", 1)])

        # Routing indexes
        await self.db.chat_routing.create_index("chat_id")

        # Intelligence metrics indexes
        await self.db.intelligence_metrics.create_index("trace_id")
        await self.db.intelligence_metrics.create_index("chat_id")
        await self.db.intelligence_metrics.create_index("user_feedback")

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT RECORDS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_record(self, message: dict[str, Any]):
        """Insert a user or assistant message."""
        assert message.get("chat_id"), "chat_id is required"

        doc = {**message}
        if "timestamp" not in doc:
            doc["timestamp"] = datetime.now(timezone.utc).isoformat()
        elif isinstance(doc["timestamp"], datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()

        await self.db.chat_records.insert_one(doc)

    async def get_chat_history(
        self,
        chat_id: str | None = None,
    ) -> list[dict] | None:
        """Get all messages for a conversation."""
        if chat_id is None:
            return None

        cursor = self.db.chat_records.find(
            {
                "chat_id": chat_id,
                "$or": [
                    {"is_streaming": {"$exists": False}},
                    {"is_streaming": False},
                ],
            },
            {"_id": 0},
        ).sort("timestamp", 1)

        return await cursor.to_list(length=1000)

    async def update_chat_record_status(
        self,
        chat_id: str,
        timestamp: Any,
        status: str,
        content: str | None = None,
        action_type: str | None = None,
    ):
        """Update a chat record's status."""
        ts = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)

        update_fields = {"status": status}
        if content is not None:
            update_fields["content"] = content
        if action_type is not None:
            update_fields["action_type"] = action_type

        await self.db.chat_records.update_one(
            {"chat_id": chat_id, "timestamp": ts},
            {"$set": update_fields},
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_metadata(self, metadata: dict[str, Any]):
        """Insert or replace chat metadata."""
        assert metadata.get("chat_id"), "chat_id is required"

        doc = {**metadata}
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()

        await self.db.chat_metadata.update_one(
            {"chat_id": doc["chat_id"]},
            {"$set": doc},
            upsert=True,
        )

    async def get_chat_metadata(self, chat_id: str) -> ChatMetadata | None:
        """Get metadata for a single chat."""
        doc = await self.db.chat_metadata.find_one(
            {"chat_id": chat_id},
            {"_id": 0},
        )
        if doc is None:
            return None
        if isinstance(doc.get("timestamp"), str):
            doc["timestamp"] = datetime.fromisoformat(doc["timestamp"])
        return ChatMetadata(**doc)

    async def get_chat_metadata_history(
        self,
        trace_id: str,
    ) -> ChatMetadataHistory:
        """Get all chat sessions for a given trace."""
        cursor = self.db.chat_metadata.find(
            {"trace_id": trace_id},
            {"_id": 0},
        )
        docs = await cursor.to_list(length=100)

        items = []
        for doc in docs:
            if isinstance(doc.get("timestamp"), str):
                doc["timestamp"] = datetime.fromisoformat(doc["timestamp"])
            items.append(ChatMetadata(**doc))

        return ChatMetadataHistory(history=items)

    # ═══════════════════════════════════════════════════════════════════════════
    # REASONING RECORDS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_reasoning_record(self, reasoning_data: dict[str, Any]):
        """Insert a reasoning/thinking chunk."""
        doc = {**reasoning_data}
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        await self.db.reasoning_records.insert_one(doc)

    async def update_reasoning_status(
        self,
        chat_id: str,
        chunk_id: int,
        status: str,
    ):
        """Update reasoning status for a chat/chunk."""
        await self.db.reasoning_records.update_many(
            {"chat_id": chat_id, "chunk_id": chunk_id},
            {"$set": {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )

    async def get_chat_reasoning(self, chat_id: str) -> list[dict]:
        """Get all reasoning chunks for a chat."""
        cursor = self.db.reasoning_records.find(
            {"chat_id": chat_id},
            {"_id": 0, "chunk_id": 1, "content": 1, "status": 1,
             "timestamp": 1, "trace_id": 1},
        ).sort([("chunk_id", 1), ("timestamp", 1)])

        return await cursor.to_list(length=1000)

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT ROUTING DECISIONS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_routing_record(self, routing_data: dict[str, Any]):
        """Insert a routing decision."""
        doc = {**routing_data}
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        await self.db.chat_routing.insert_one(doc)

    async def get_routing_history(self, chat_id: str) -> list[dict]:
        """Get all routing decisions for a chat."""
        cursor = self.db.chat_routing.find(
            {"chat_id": chat_id},
            {"_id": 0},
        ).sort("timestamp", 1)
        return await cursor.to_list(length=100)

    # ═══════════════════════════════════════════════════════════════════════════
    # INTELLIGENCE METRICS (for evaluation loop)
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_intelligence_metrics(self, metrics: dict[str, Any]):
        """Store intelligence metrics after each RCA query.

        MongoDB stores pattern_matches and ranked_causes as native
        JSON arrays — no serialization needed (unlike SQLite).
        """
        doc = {**metrics}
        if isinstance(doc.get("timestamp"), datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()

        await self.db.intelligence_metrics.insert_one(doc)

    async def get_intelligence_metrics(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Query intelligence metrics for the evaluation loop.

        Supports filtering by:
            trace_id, chat_id, pattern_name, user_feedback, fast_path_used, limit
        """
        query = {}

        if filters:
            if "trace_id" in filters:
                query["trace_id"] = filters["trace_id"]

            if "chat_id" in filters:
                query["chat_id"] = filters["chat_id"]

            if "pattern_name" in filters:
                # MongoDB can search inside arrays natively
                query["pattern_matches.name"] = filters["pattern_name"]

            if "user_feedback" in filters:
                if filters["user_feedback"] is None:
                    query["user_feedback"] = None
                else:
                    query["user_feedback"] = filters["user_feedback"]

            if "fast_path_used" in filters:
                query["fast_path_used"] = filters["fast_path_used"]

        limit = filters.get("limit", 100) if filters else 100

        cursor = self.db.intelligence_metrics.find(
            query,
            {"_id": 0},
        ).sort("timestamp", -1).limit(limit)

        return await cursor.to_list(length=limit)

    async def update_user_feedback(
        self,
        chat_id: str,
        trace_id: str,
        feedback: str,
        comment: str | None = None,
    ):
        """Update user feedback for a specific RCA result."""
        await self.db.intelligence_metrics.update_one(
            {"chat_id": chat_id, "trace_id": trace_id},
            {"$set": {
                "user_feedback": feedback,
                "feedback_timestamp": datetime.now(timezone.utc).isoformat(),
                "feedback_comment": comment,
            }},
        )

    async def get_pattern_accuracy(self, pattern_name: str) -> dict:
        """Get accuracy stats for a specific pattern."""
        pipeline = [
            {"$match": {"pattern_matches.name": pattern_name}},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "positive": {
                    "$sum": {"$cond": [
                        {"$eq": ["$user_feedback", "positive"]}, 1, 0
                    ]}
                },
                "negative": {
                    "$sum": {"$cond": [
                        {"$eq": ["$user_feedback", "negative"]}, 1, 0
                    ]}
                },
                "unrated": {
                    "$sum": {"$cond": [
                        {"$eq": ["$user_feedback", None]}, 1, 0
                    ]}
                },
                "avg_processing_time": {"$avg": "$processing_time_ms"},
            }},
        ]

        results = await self.db.intelligence_metrics.aggregate(pipeline).to_list(1)

        if not results:
            return {
                "pattern_name": pattern_name,
                "total": 0, "positive": 0, "negative": 0,
                "unrated": 0, "accuracy": 0.0,
                "avg_processing_time_ms": None,
            }

        r = results[0]
        rated = r["positive"] + r["negative"]
        return {
            "pattern_name": pattern_name,
            "total": r["total"],
            "positive": r["positive"],
            "negative": r["negative"],
            "unrated": r["unrated"],
            "accuracy": r["positive"] / max(rated, 1),
            "avg_processing_time_ms": r["avg_processing_time"],
        }
