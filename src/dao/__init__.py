"""
Shared Pydantic models for all database records.
Both SQLite and MongoDB DAOs use these same types.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class ChatRecord(BaseModel):
    """A single chat message (user or assistant)."""
    chat_id: str
    timestamp: str | datetime
    role: str                              # "user" or "assistant"
    content: str                           # The actual message text
    user_content: Optional[str] = None     # Original user message (if assistant)
    trace_id: Optional[str] = None
    span_ids: Optional[list[str]] = None
    start_time: Optional[str | datetime] = None
    end_time: Optional[str | datetime] = None
    model: Optional[str] = None            # LLM model used (e.g., "llama-3.3-70b-versatile")
    mode: Optional[str] = None             # "CHAT" or "AGENT"
    message_type: Optional[str] = None     # "user" or "assistant"
    chunk_id: Optional[int] = None
    action_type: Optional[str] = None      # "AGENT_CHAT", "GENERAL", etc.
    status: Optional[str] = None           # "SUCCESS", "ERROR", etc.
    user_message: Optional[str] = None
    context: Optional[str] = None          # Compressed context sent to LLM
    reference: Optional[list[dict]] = None # Evidence references
    is_streaming: Optional[bool] = None
    stream_update: Optional[bool] = None


class ChatMetadata(BaseModel):
    """Metadata for a chat session (for listing past chats)."""
    chat_id: str
    timestamp: str | datetime
    chat_title: str
    trace_id: str
    user_id: Optional[str] = None


class ChatMetadataHistory(BaseModel):
    """List of chat metadata entries."""
    history: list[ChatMetadata] = Field(default_factory=list)


class ReasoningRecord(BaseModel):
    """LLM thinking/reasoning chunk (for streaming display)."""
    chat_id: str
    chunk_id: int
    content: str
    status: str                            # "pending", "complete", "error"
    timestamp: str | datetime
    trace_id: Optional[str] = None
    updated_at: Optional[str] = None


class RoutingRecord(BaseModel):
    """Agent routing decision record."""
    chat_id: str
    timestamp: str | datetime
    user_message: Optional[str] = None
    agent_type: str                        # "single_rca", "general", "code"
    reasoning: Optional[str] = None        # Why this agent was chosen
    chat_mode: Optional[str] = None        # "CHAT" or "AGENT"
    trace_id: Optional[str] = None
    user_sub: Optional[str] = None


class IntelligenceMetrics(BaseModel):
    """Intelligence Layer metrics — for evaluation loop and improvement.

    Stored after every RCA query. user_feedback is NULL initially
    and updated later when the user clicks thumbs up/down.
    """
    trace_id: str
    chat_id: str
    timestamp: str | datetime

    # Intelligence Layer outputs
    pattern_matches: list[dict] = Field(
        default_factory=list,
        description="[{'name': 'slow_query', 'confidence': 0.9, 'category': 'db'}]"
    )
    ranked_causes: list[dict] = Field(
        default_factory=list,
        description="[{'span_function': 'db_query', 'score': 0.95, 'rank': 1}]"
    )
    fast_path_used: bool = False
    compression_ratio: Optional[float] = None
    processing_time_ms: Optional[float] = None

    # Validation results (from synthesizer)
    validation_result: Optional[dict] = Field(
        default=None,
        description="{'confidence': 0.92, 'passed': true, 'issues': 0, 'fallback_used': false}"
    )

    # User feedback (filled later)
    user_feedback: Optional[str] = None        # "positive", "negative", or NULL
    feedback_timestamp: Optional[str] = None
    feedback_comment: Optional[str] = None
