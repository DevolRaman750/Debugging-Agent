"""HTTP router layer for chat endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from config import FeedbackRequest
from driver.chat_logic import ChatLogic


def get_user_credentials(request: Request) -> tuple[str, str | None, str]:
    """Extract user credentials from trusted proxy headers.

    Returns:
        A tuple of (user_email, bearer_token, user_sub).

    Raises:
        ValueError: If required identity headers are missing.
    """
    user_email = request.headers.get("X-User-Email", "").strip()
    user_sub = request.headers.get("X-User-Sub", "").strip()
    auth_header = request.headers.get("Authorization", "")
    bearer_token = auth_header.replace("Bearer ", "", 1) if auth_header.startswith("Bearer ") else None

    if not user_email or not user_sub:
        raise ValueError("Missing authenticated user headers")

    return user_email, bearer_token, user_sub


class _NoopLimiter:
    """Fallback limiter used when a real limiter is not wired in this project."""

    def limit(self, _rule: str):
        def decorator(func):
            return func

        return decorator


class ChatRouterClass:
    """Router layer handling HTTP concerns for chat endpoints."""

    def __init__(self, driver: ChatLogic | None = None):
        self.driver = driver or ChatLogic()
        self.logger = __import__("logging").getLogger(__name__)
        self.limiter = _NoopLimiter()
        self.router = APIRouter(prefix="/v1/explore", tags=["chat"])
        self._register_routes()

    def _register_routes(self) -> None:
        """Register route handlers on the APIRouter instance."""

        @self.router.post("/update-feedback")
        @self.limiter.limit("60/minute")
        async def update_feedback_endpoint(
            request: Request,
            req_data: FeedbackRequest,
        ) -> dict[str, Any]:
            return await self.update_feedback(request, req_data)

    async def update_feedback(
        self,
        request: Request,
        req_data: FeedbackRequest,
    ) -> dict[str, Any]:
        """Handle feedback update requests and map exceptions to HTTP errors."""
        try:
            user_email, _, user_sub = get_user_credentials(request)
            self.logger.debug(
                "Feedback update request from %s (sub=%s) for chat_id=%s",
                user_email,
                user_sub,
                req_data.chat_id,
            )

            result = await self.driver.update_feedback(
                chat_id=req_data.chat_id,
                message_timestamp=req_data.message_timestamp,
                feedback=req_data.feedback,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            self.logger.error("Error updating feedback: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to update feedback: {str(e)}",
            ) from e
