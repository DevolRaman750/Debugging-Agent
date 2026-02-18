from abc import ABC, abstractmethod
from datetime import datetime
from src.routing.types import ChatbotResponse, ChatModel
from src.context.model import SpanNode
class BaseAgent(ABC):
    """Base class for all agents."""
    
    @abstractmethod
    async def chat(
        self,
        trace_id: str,
        chat_id: str,
        user_message: str,
        model: ChatModel,
        timestamp: datetime,
        tree: SpanNode | None = None,
        chat_history: list[dict] | None = None,
        **kwargs
    ) -> ChatbotResponse:
        """Main chat method - must be implemented by subclasses."""
        pass