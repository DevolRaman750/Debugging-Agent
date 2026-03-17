"""Shared request/response models for API routing."""

from typing import Literal

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    """Incoming feedback payload for rating an AI response."""

    chat_id: str
    message_timestamp: float
    feedback: Literal["positive", "negative"]
