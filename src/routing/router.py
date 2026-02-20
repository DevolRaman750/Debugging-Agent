from typing import Literal
from openai import AsyncOpenAI  # Same library — Groq is OpenAI-compatible
from src.routing.types import ChatMode, RouterOutput
from src.routing.prompts.router_prompts import ROUTER_SYSTEM_PROMPT
from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


GROQ_DEFAULT_MODEL = GROQ_MODEL


class ChatRouter:
    """Routes user queries to appropriate agents using Groq LLM."""

    def __init__(self, client: AsyncOpenAI = None, groq_api_key: str = None):
        """
        Initialize router.

        Args:
            client: Pre-configured AsyncOpenAI client (pointed at Groq)
            groq_api_key: Groq API key (free from console.groq.com)
        """
        api_key = groq_api_key or GROQ_API_KEY
        if client:
            self.client = client
        elif api_key:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=GROQ_BASE_URL
            )
        else:
            self.client = None

        self.system_prompt = ROUTER_SYSTEM_PROMPT

    async def route_query(
        self,
        user_message: str,
        chat_mode: ChatMode,
        model: str = GROQ_DEFAULT_MODEL,  # Changed from "gpt-4o"
        groq_api_key: str | None = None,
        has_trace_context: bool = False,
        is_github_issue: bool = False,
        is_github_pr: bool = False,
        source_code_related: bool = False,
    ) -> RouterOutput:
        """Route a user query to the appropriate agent."""

        # Use provided key or default client
        if groq_api_key:
            client = AsyncOpenAI(
                api_key=groq_api_key,
                base_url=GROQ_BASE_URL  # ← Points to Groq, not OpenAI
            )
        elif self.client:
            client = self.client
        else:
            # Fallback without LLM
            if has_trace_context:
                return RouterOutput(
                    agent_type="single_rca",
                    reasoning="Default to RCA agent with trace context"
                )
            return RouterOutput(
                agent_type="general",
                reasoning="Default to general agent without context"
            )

        # Determine allowed agents based on mode
        if chat_mode == ChatMode.CHAT:
            allowed = "Available: 'single_rca', 'general'. (code not allowed in CHAT mode)"

            def route_fn(agent_type: Literal["single_rca", "general"], reasoning: str):
                return {"agent_type": agent_type, "reasoning": reasoning}
        else:
            allowed = "Available: 'single_rca', 'code', 'general'"

            def route_fn(agent_type: Literal["single_rca", "code", "general"], reasoning: str):
                return {"agent_type": agent_type, "reasoning": reasoning}

        # Build routing request
        user_content = f"""User message: {user_message}

Context:
- Has trace/logs available: {has_trace_context}
- GitHub issue creation detected: {is_github_issue}
- GitHub PR creation detected: {is_github_pr}
- Source code related: {source_code_related}
- Chat mode: {chat_mode.value}
{allowed}
Which agent should handle this query?"""

        try:
            response = await client.chat.completions.create(
                model=model,  # Will use "llama-3.3-70b-versatile"
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "route_to_agent",
                        "description": "Route query to the appropriate agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "agent_type": {
                                    "type": "string",
                                    "enum": ["single_rca", "code", "general"]
                                },
                                "reasoning": {"type": "string"}
                            },
                            "required": ["agent_type", "reasoning"]
                        }
                    }
                }],
                temperature=0.3
            )

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                import json
                args = json.loads(tool_calls[0].function.arguments)

                # Validate mode restrictions
                if chat_mode == ChatMode.CHAT and args["agent_type"] == "code":
                    return RouterOutput(
                        agent_type="single_rca",
                        reasoning="Code agent not allowed in CHAT mode, using single_rca"
                    )

                return RouterOutput(**args)

        except Exception as e:
            print(f"Routing error: {e}")

        # Fallback
        return RouterOutput(
            agent_type="single_rca" if has_trace_context else "general",
            reasoning="Fallback routing"
        )