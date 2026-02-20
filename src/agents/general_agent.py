from datetime import datetime
from openai import AsyncOpenAI
from src.agents.base import BaseAgent
from src.routing.types import ChatbotResponse, ChatModel, MessageType
from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

GENERAL_SYSTEM_PROMPT = """You are TraceRoot, a helpful AI assistant.
You can answer general questions about:
- Software development and debugging concepts
- Distributed systems and observability
- Best practices for logging and tracing
- General programming questions
Be helpful, concise, and accurate."""

class GeneralAgent(BaseAgent):
    """General purpose agent for non-trace queries."""
    
    def __init__(self, groq_api_key: str = None):
        api_key = groq_api_key or GROQ_API_KEY
        if api_key:
            self.chat_client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        else:
            self.chat_client = None
        self.system_prompt = GENERAL_SYSTEM_PROMPT
    
    async def chat(
        self,
        trace_id: str,
        chat_id: str,
        user_message: str,
        model: ChatModel,
        timestamp: datetime,
        chat_history: list[dict] | None = None,
        groq_api_key: str | None = None,
        **kwargs
    ) -> ChatbotResponse:
        """Simple chat without trace context."""
        
        if groq_api_key:
            client = AsyncOpenAI(api_key=groq_api_key, base_url=GROQ_BASE_URL)
        elif self.chat_client:
            client = self.chat_client
        else:
            return ChatbotResponse(
                time=datetime.now(),
                message="Error: No LLM client configured.",
                reference=[],
                message_type=MessageType.ASSISTANT,
                chat_id=chat_id,
            )
        
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add history
        if chat_history:
            for h in chat_history[-10:]:
                messages.append({"role": h["role"], "content": h["content"]})
        
        messages.append({"role": "user", "content": user_message})
        
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages
        )
        
        return ChatbotResponse(
            time=datetime.now(),
            message=response.choices[0].message.content,
            reference=[],
            message_type=MessageType.ASSISTANT,
            chat_id=chat_id
        )