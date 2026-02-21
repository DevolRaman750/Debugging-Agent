"""
SQLite Persistence Layer for TraceRoot
========================================
Local development database using aiosqlite.

5 Tables:
  1. chat_records          — Every user/assistant message
  2. chat_metadata         — Chat session index (titles, trace links)
  3. reasoning_records     — LLM thinking/reasoning chunks (streaming)
  4. chat_routing          — Agent routing decisions
  5. intelligence_metrics  — Intel Layer results + validation + user feedback
"""

import json
import os
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from src.dao.types import ChatMetadata, ChatMetadataHistory

DB_PATH = os.getenv("SQLITE_DB_PATH", "traceroot.db")


class TraceRootSQLiteClient:

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ═══════════════════════════════════════════════════════════════════════════
    # DATABASE INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════

    async def _init_db(self):
        """Create all 5 tables and indexes if they don't exist."""
        async with aiosqlite.connect(self.db_path) as db:

            # ── Table 1: Chat Records ──
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_content TEXT,
                    trace_id TEXT,
                    span_ids TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    model TEXT,
                    mode TEXT,
                    message_type TEXT,
                    chunk_id INTEGER,
                    action_type TEXT,
                    status TEXT,
                    user_message TEXT,
                    context TEXT,
                    reference TEXT,
                    is_streaming BOOLEAN,
                    stream_update BOOLEAN
                )
            """)

            # ── Table 2: Chat Metadata ──
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    chat_title TEXT NOT NULL,
                    trace_id TEXT NOT NULL
                )
            """)

            # ── Table 3: Reasoning Records ──
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reasoning_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    trace_id TEXT,
                    updated_at TEXT
                )
            """)

            # ── Table 4: Chat Routing Decisions ──
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_routing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_message TEXT,
                    agent_type TEXT NOT NULL,
                    reasoning TEXT,
                    chat_mode TEXT,
                    trace_id TEXT,
                    user_sub TEXT
                )
            """)

            # ── Table 5: Intelligence Metrics (NEW) ──
            await db.execute("""
                CREATE TABLE IF NOT EXISTS intelligence_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    pattern_matches TEXT,
                    ranked_causes TEXT,
                    fast_path_used BOOLEAN DEFAULT 0,
                    compression_ratio REAL,
                    processing_time_ms REAL,
                    validation_result TEXT,
                    user_feedback TEXT,
                    feedback_timestamp TEXT,
                    feedback_comment TEXT
                )
            """)

            # ── Indexes ──
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_records_chat_id "
                "ON chat_records(chat_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_records_timestamp "
                "ON chat_records(timestamp)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_metadata_trace_id "
                "ON chat_metadata(trace_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_chat_id "
                "ON reasoning_records(chat_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_reasoning_chunk "
                "ON reasoning_records(chat_id, chunk_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_routing_chat_id "
                "ON chat_routing(chat_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_intel_trace "
                "ON intelligence_metrics(trace_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_intel_chat "
                "ON intelligence_metrics(chat_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_intel_feedback "
                "ON intelligence_metrics(user_feedback)"
            )

            await db.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT RECORDS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_record(self, message: dict[str, Any]):
        """Insert a user or assistant message.

        Args:
            message: Dict with at minimum: chat_id, role, content.
                     Optional: timestamp, trace_id, reference, model, etc.
        """
        assert message.get("chat_id"), "chat_id is required"

        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            timestamp = self._to_iso(message.get(
                "timestamp", datetime.now(timezone.utc)
            ))

            # Serialize complex fields to JSON
            span_ids = self._to_json(message.get("span_ids"))
            reference = self._to_json(message.get("reference"))
            start_time = self._to_iso(message.get("start_time"))
            end_time = self._to_iso(message.get("end_time"))

            await db.execute(
                """INSERT INTO chat_records (
                    chat_id, timestamp, role, content, user_content,
                    trace_id, span_ids, start_time, end_time, model,
                    mode, message_type, chunk_id, action_type, status,
                    user_message, context, reference, is_streaming, stream_update
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )""",
                (
                    message["chat_id"],
                    timestamp,
                    message.get("role", ""),
                    message.get("content", ""),
                    message.get("user_content"),
                    message.get("trace_id"),
                    span_ids,
                    start_time,
                    end_time,
                    message.get("model"),
                    message.get("mode"),
                    message.get("message_type"),
                    message.get("chunk_id"),
                    message.get("action_type"),
                    message.get("status"),
                    message.get("user_message"),
                    message.get("context"),
                    reference,
                    message.get("is_streaming"),
                    message.get("stream_update"),
                )
            )
            await db.commit()

    async def get_chat_history(
        self,
        chat_id: str | None = None,
    ) -> list[dict] | None:
        """Get all messages for a conversation, ordered by time."""
        if chat_id is None:
            return None

        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM chat_records
                   WHERE chat_id = ?
                   AND (is_streaming IS NULL OR is_streaming = 0)
                   AND (stream_update IS NULL OR stream_update = 0)
                   ORDER BY timestamp ASC""",
                (chat_id,)
            )
            rows = await cursor.fetchall()

            items = []
            for row in rows:
                item = dict(row)
                # Deserialize JSON fields
                if item.get("span_ids"):
                    item["span_ids"] = json.loads(item["span_ids"])
                if item.get("reference"):
                    item["reference"] = json.loads(item["reference"])
                items.append(item)

            return items

    async def update_chat_record_status(
        self,
        chat_id: str,
        timestamp: Any,
        status: str,
        content: str | None = None,
        action_type: str | None = None,
    ):
        """Update a chat record's status and optionally content/action_type."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            ts = self._to_iso(timestamp)
            if content and action_type:
                await db.execute(
                    """UPDATE chat_records
                       SET status = ?, content = ?, action_type = ?
                       WHERE chat_id = ? AND timestamp = ?""",
                    (status, content, action_type, chat_id, ts)
                )
            elif content:
                await db.execute(
                    """UPDATE chat_records
                       SET status = ?, content = ?
                       WHERE chat_id = ? AND timestamp = ?""",
                    (status, content, chat_id, ts)
                )
            else:
                await db.execute(
                    """UPDATE chat_records
                       SET status = ?
                       WHERE chat_id = ? AND timestamp = ?""",
                    (status, chat_id, ts)
                )
            await db.commit()

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT METADATA
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_metadata(self, metadata: dict[str, Any]):
        """Insert or replace chat metadata (title, trace link)."""
        assert metadata.get("chat_id"), "chat_id is required"

        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            timestamp = self._to_iso(metadata.get(
                "timestamp", datetime.now(timezone.utc)
            ))

            await db.execute(
                """INSERT OR REPLACE INTO chat_metadata (
                    chat_id, timestamp, chat_title, trace_id
                ) VALUES (?, ?, ?, ?)""",
                (
                    metadata["chat_id"],
                    timestamp,
                    metadata.get("chat_title", ""),
                    metadata.get("trace_id", ""),
                )
            )
            await db.commit()

    async def get_chat_metadata(self, chat_id: str) -> ChatMetadata | None:
        """Get metadata for a single chat."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM chat_metadata WHERE chat_id = ?",
                (chat_id,)
            )
            row = await cursor.fetchone()

            if row is None:
                return None

            item = dict(row)
            if item.get("timestamp"):
                item["timestamp"] = datetime.fromisoformat(item["timestamp"])
            return ChatMetadata(**item)

    async def get_chat_metadata_history(
        self,
        trace_id: str,
    ) -> ChatMetadataHistory:
        """Get all chat sessions for a given trace."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM chat_metadata WHERE trace_id = ?",
                (trace_id,)
            )
            rows = await cursor.fetchall()

            items = []
            for row in rows:
                item = dict(row)
                if item.get("timestamp"):
                    item["timestamp"] = datetime.fromisoformat(item["timestamp"])
                items.append(ChatMetadata(**item))

            return ChatMetadataHistory(history=items)

    # ═══════════════════════════════════════════════════════════════════════════
    # REASONING RECORDS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_reasoning_record(self, reasoning_data: dict[str, Any]):
        """Insert a reasoning/thinking chunk."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            timestamp = self._to_iso(reasoning_data.get(
                "timestamp", datetime.now(timezone.utc)
            ))

            await db.execute(
                """INSERT INTO reasoning_records (
                    chat_id, chunk_id, content, status, timestamp, trace_id
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    reasoning_data["chat_id"],
                    reasoning_data["chunk_id"],
                    reasoning_data["content"],
                    reasoning_data["status"],
                    timestamp,
                    reasoning_data.get("trace_id"),
                )
            )
            await db.commit()

    async def update_reasoning_status(
        self,
        chat_id: str,
        chunk_id: int,
        status: str,
    ):
        """Update reasoning status for a chat/chunk."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            updated_at = datetime.now(timezone.utc).isoformat()
            await db.execute(
                """UPDATE reasoning_records
                   SET status = ?, updated_at = ?
                   WHERE chat_id = ? AND chunk_id = ?""",
                (status, updated_at, chat_id, chunk_id)
            )
            await db.commit()

    async def get_chat_reasoning(self, chat_id: str) -> list[dict]:
        """Get all reasoning chunks for a chat."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT chunk_id, content, status, timestamp, trace_id
                   FROM reasoning_records
                   WHERE chat_id = ?
                   ORDER BY chunk_id ASC, timestamp ASC""",
                (chat_id,)
            )
            rows = await cursor.fetchall()

            return [
                {
                    "chunk_id": dict(row)["chunk_id"] or 0,
                    "content": dict(row)["content"] or "",
                    "status": dict(row)["status"] or "pending",
                    "timestamp": dict(row)["timestamp"],
                    "trace_id": dict(row)["trace_id"],
                }
                for row in rows
            ]

    # ═══════════════════════════════════════════════════════════════════════════
    # CHAT ROUTING DECISIONS
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_chat_routing_record(self, routing_data: dict[str, Any]):
        """Insert a routing decision (which agent was chosen and why)."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            timestamp = self._to_iso(routing_data.get(
                "timestamp", datetime.now(timezone.utc)
            ))

            await db.execute(
                """INSERT INTO chat_routing (
                    chat_id, timestamp, user_message, agent_type,
                    reasoning, chat_mode, trace_id, user_sub
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    routing_data["chat_id"],
                    timestamp,
                    routing_data.get("user_message"),
                    routing_data["agent_type"],
                    routing_data.get("reasoning"),
                    routing_data.get("chat_mode"),
                    routing_data.get("trace_id"),
                    routing_data.get("user_sub"),
                )
            )
            await db.commit()

    async def get_routing_history(self, chat_id: str) -> list[dict]:
        """Get all routing decisions for a chat."""
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT * FROM chat_routing
                   WHERE chat_id = ?
                   ORDER BY timestamp ASC""",
                (chat_id,)
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ═══════════════════════════════════════════════════════════════════════════
    # INTELLIGENCE METRICS (NEW — for evaluation loop)
    # ═══════════════════════════════════════════════════════════════════════════

    async def insert_intelligence_metrics(self, metrics: dict[str, Any]):
        """Store intelligence metrics after each RCA query.

        Called by SingleRCAAgent after pipeline + validation complete.
        The user_feedback field starts as NULL and is updated later
        when the user clicks thumbs up/down.

        Args:
            metrics: Dict with keys:
                trace_id, chat_id, timestamp,
                pattern_matches (list[dict]), ranked_causes (list[dict]),
                fast_path_used (bool), compression_ratio (float),
                processing_time_ms (float), validation_result (dict),
                user_feedback (None initially)
        """
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            timestamp = self._to_iso(metrics.get(
                "timestamp", datetime.now(timezone.utc)
            ))

            await db.execute(
                """INSERT INTO intelligence_metrics (
                    trace_id, chat_id, timestamp,
                    pattern_matches, ranked_causes,
                    fast_path_used, compression_ratio, processing_time_ms,
                    validation_result, user_feedback,
                    feedback_timestamp, feedback_comment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    metrics["trace_id"],
                    metrics["chat_id"],
                    timestamp,
                    self._to_json(metrics.get("pattern_matches")),
                    self._to_json(metrics.get("ranked_causes")),
                    1 if metrics.get("fast_path_used") else 0,
                    metrics.get("compression_ratio"),
                    metrics.get("processing_time_ms"),
                    self._to_json(metrics.get("validation_result")),
                    metrics.get("user_feedback"),
                    None,  # feedback_timestamp — set later
                    None,  # feedback_comment — set later
                )
            )
            await db.commit()

    async def get_intelligence_metrics(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[dict]:
        """Query intelligence metrics for the evaluation loop.

        Supports filtering by:
            trace_id      — metrics for a specific trace
            chat_id       — metrics for a specific conversation
            pattern_name  — all records where a pattern was matched
            user_feedback — "positive", "negative", or None (unfeedback'd)
            fast_path_used — True/False
            limit         — max records to return (default 100)

        Returns:
            List of metric dicts with JSON fields deserialized.
        """
        await self._init_db()

        query = "SELECT * FROM intelligence_metrics"
        conditions = []
        params = []

        if filters:
            if "trace_id" in filters:
                conditions.append("trace_id = ?")
                params.append(filters["trace_id"])

            if "chat_id" in filters:
                conditions.append("chat_id = ?")
                params.append(filters["chat_id"])

            if "pattern_name" in filters:
                # Search within the JSON string for the pattern name
                conditions.append("pattern_matches LIKE ?")
                params.append(f'%"{filters["pattern_name"]}"%')

            if "user_feedback" in filters:
                if filters["user_feedback"] is None:
                    conditions.append("user_feedback IS NULL")
                else:
                    conditions.append("user_feedback = ?")
                    params.append(filters["user_feedback"])

            if "fast_path_used" in filters:
                conditions.append("fast_path_used = ?")
                params.append(1 if filters["fast_path_used"] else 0)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY timestamp DESC"

        limit = filters.get("limit", 100) if filters else 100
        query += f" LIMIT {int(limit)}"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                item = dict(row)
                # Deserialize JSON fields
                if item.get("pattern_matches"):
                    item["pattern_matches"] = json.loads(item["pattern_matches"])
                if item.get("ranked_causes"):
                    item["ranked_causes"] = json.loads(item["ranked_causes"])
                if item.get("validation_result"):
                    item["validation_result"] = json.loads(item["validation_result"])
                item["fast_path_used"] = bool(item.get("fast_path_used"))
                results.append(item)

            return results

    async def update_user_feedback(
        self,
        chat_id: str,
        trace_id: str,
        feedback: str,
        comment: str | None = None,
    ):
        """Update user feedback for a specific RCA result.

        Called when the user clicks thumbs up or thumbs down.

        Args:
            chat_id: The conversation ID
            trace_id: The trace ID
            feedback: "positive" or "negative"
            comment: Optional text comment from the user
        """
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            feedback_ts = datetime.now(timezone.utc).isoformat()

            await db.execute(
                """UPDATE intelligence_metrics
                   SET user_feedback = ?,
                       feedback_timestamp = ?,
                       feedback_comment = ?
                   WHERE chat_id = ? AND trace_id = ?""",
                (feedback, feedback_ts, comment, chat_id, trace_id)
            )
            await db.commit()

    async def get_pattern_accuracy(self, pattern_name: str) -> dict:
        """Get accuracy stats for a specific pattern (for evaluation loop).

        Returns:
            Dict with total, positive, negative, unrated counts and accuracy %.
        """
        await self._init_db()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN user_feedback = 'positive' THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN user_feedback = 'negative' THEN 1 ELSE 0 END) as negative,
                    SUM(CASE WHEN user_feedback IS NULL THEN 1 ELSE 0 END) as unrated,
                    AVG(processing_time_ms) as avg_processing_time
                   FROM intelligence_metrics
                   WHERE pattern_matches LIKE ?""",
                (f'%"{pattern_name}"%',)
            )
            row = await cursor.fetchone()

            total = row[0] or 0
            positive = row[1] or 0
            negative = row[2] or 0

            return {
                "pattern_name": pattern_name,
                "total": total,
                "positive": positive,
                "negative": negative,
                "unrated": row[3] or 0,
                "accuracy": positive / max(positive + negative, 1),
                "avg_processing_time_ms": row[4],
            }

    # ═══════════════════════════════════════════════════════════════════════════
    # UTILITY HELPERS
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_iso(value: Any) -> str | None:
        """Convert datetime to ISO string, or return string as-is."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    @staticmethod
    def _to_json(value: Any) -> str | None:
        """Convert list/dict to JSON string for SQLite storage."""
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value, default=str)
        if isinstance(value, str):
            return value  # Already JSON string
        return json.dumps(value, default=str)
