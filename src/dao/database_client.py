"""Database facade used by driver layer.

This wrapper picks the configured backend DAO (SQLite or MongoDB)
and exposes stable methods used by business logic.
"""

from src.dao.factory import create_dao


class DatabaseClient:
    """Facade over the configured persistence backend."""

    def __init__(self):
        self._dao = create_dao()

    async def update_user_feedback(
        self,
        chat_id: str,
        timestamp: float,
        feedback: str,
    ) -> bool:
        """Update user feedback on intelligence metrics.

        Args:
            chat_id: Conversation identifier.
            timestamp: Epoch timestamp of the rated assistant response.
            feedback: Either "positive" or "negative".

        Returns:
            True when at least one record was updated, otherwise False.
        """
        return await self._dao.update_user_feedback(
            chat_id=chat_id,
            timestamp=timestamp,
            feedback=feedback,
        )
