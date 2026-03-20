"""
Agent types - Enums and models used across filter, chunk, and summarizer.
Adapted from rest/agent/typing.py and rest/agent/output/
"""

from enum import Enum
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENUMS - What fields to include in the tree output
# ═══════════════════════════════════════════════════════════════════════════════

class LogFeature(Enum):
    """Which log fields to include when serializing the tree.
    
    The LLM decides which of these are relevant to the user's question.
    For example, if user asks "show me errors", the LLM picks LOG_LEVEL
    and LOG_MESSAGE_VALUE (no need for line numbers or timestamps).
    """
    LOG_UTC_TIMESTAMP = "log utc timestamp"
    LOG_LEVEL = "log level"
    LOG_FILE_NAME = "file name"
    LOG_FUNC_NAME = "function name"
    LOG_MESSAGE_VALUE = "log message value"
    LOG_LINE_NUMBER = "line number"
    LOG_SOURCE_CODE_LINE = "log line source code"


class SpanFeature(Enum):
    """Which span fields to include when serializing the tree.
    
    For example, if user asks "why is this slow?", the LLM picks
    SPAN_LATENCY. If they ask "what happened at 3pm?", it picks timestamps.
    """
    SPAN_LATENCY = "span latency"
    SPAN_UTC_START_TIME = "span utc start time"
    SPAN_UTC_END_TIME = "span utc end time"


class FeatureOps(Enum):
    """Operations for filtering log nodes."""
    EQUAL = "equal"
    NOT_EQUAL = "not equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not contains"


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT MODELS - Structured responses from the LLM
# ═══════════════════════════════════════════════════════════════════════════════

class Reference(BaseModel):
    """Reference to evidence in the trace."""
    type: str = Field(description="Type: 'span', 'log', or 'code'")
    number: int | None = Field(default=None, description="Reference number [1], [2], etc.")
    span_id: str | None = Field(default=None, description="Span ID if type is 'span'")
    span_function_name: str | None = Field(default=None, description="Span function/operation name")
    log_message: str | None = Field(default=None, description="Log message if type is 'log'")
    line_number: int | None = Field(default=None, description="Line number if type is 'code'")
    url: str | None = Field(default=None, description="URL if applicable")


class ChatOutput(BaseModel):
    """Structured output from the RCA agent."""
    answer: str = Field(
        description=(
            "The main response or answer to the user's query based on given context. "
            "Include reference numbers like [1], [2] at the end of relevant lines."
        )
    )
    reference: list[Reference] = Field(
        default=[],
        description="References to spans, logs, and source code supporting the answer."
    )
