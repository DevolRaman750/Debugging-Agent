"""Business logic layer for chat-related API operations."""

from typing import Any

from src.dao.database_client import DatabaseClient


class ChatLogic:
    """Driver layer that orchestrates feedback updates through the DAO."""

    def __init__(self, db_client: DatabaseClient | None = None):
        self.db_client = db_client or DatabaseClient()

    async def update_feedback(
        self,
        chat_id: str,
        message_timestamp: float,
        feedback: str,
    ) -> dict[str, Any]:
        """Persist user feedback for a specific chat response.

        Args:
            chat_id: Conversation identifier.
            message_timestamp: Epoch timestamp for the rated assistant message.
            feedback: Either "positive" or "negative".

        Returns:
            A success payload when persistence succeeds.

        Raises:
            ValueError: If feedback is invalid or no matching record is found.
        """
        if feedback not in {"positive", "negative"}:
            raise ValueError("feedback must be 'positive' or 'negative'")

        updated = await self.db_client.update_user_feedback(
            chat_id=chat_id,
            timestamp=message_timestamp,
            feedback=feedback,
        )

        if not updated:
            raise ValueError("No matching intelligence metrics record found for feedback update")

        return {"success": True, "message": "Feedback updated successfully"}
