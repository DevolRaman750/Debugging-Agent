import asyncio
from datetime import datetime
from src.routing.router import ChatRouter
from src.routing.types import ChatRequest, ChatbotResponse, ChatMode
from src.agents.single_rca_agent import SingleRCAAgent
from src.agents.general_agent import GeneralAgent
# from src.agents.code_agent import CodeAgent  # For GitHub operations
from src.service.provider import ObservabilityProvider
from src.context.tree_builder import build_heterogeneous_tree
from src.context.utils import find_root_span
class ChatLogic:
    """Main orchestrator for chat operations."""
    
    def __init__(self):
        self.chat_router = ChatRouter()
        self.single_rca_agent = SingleRCAAgent()
        self.general_agent = GeneralAgent()
        # self.code_agent = CodeAgent()
        self.observe_provider = ObservabilityProvider.create_jaeger_provider()
    
    async def post_chat(
        self,
        req_data: ChatRequest,
        openai_token: str | None = None,
    ) -> ChatbotResponse:
        """Main chat entry point."""
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Route Query
        # ═══════════════════════════════════════════════════════════════════════
        router_output = await self.chat_router.route_query(
            user_message=req_data.message,
            chat_mode=req_data.mode,
            openai_token=openai_token,
            has_trace_context=bool(req_data.trace_id),
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Handle General Agent (no trace needed)
        # ═══════════════════════════════════════════════════════════════════════
        if router_output.agent_type == "general":
            return await self.general_agent.chat(
                trace_id=req_data.trace_id or "",
                chat_id=req_data.chat_id,
                user_message=req_data.message,
                model=req_data.model,
                timestamp=req_data.time,
                openai_token=openai_token,
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Fetch Telemetry (for trace-based agents)
        # ═══════════════════════════════════════════════════════════════════════
        trace = await self.observe_provider.trace_client.get_trace_by_id(
            req_data.trace_id
        )
        
        if not trace:
            return ChatbotResponse(
                time=datetime.now(),
                message=f"Trace {req_data.trace_id} not found",
                reference=[],
                chat_id=req_data.chat_id
            )
        
        logs = await self.observe_provider.log_client.get_logs_by_trace_id(
            req_data.trace_id
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Build Tree
        # ═══════════════════════════════════════════════════════════════════════
        root_span = find_root_span(trace.spans)
        tree = build_heterogeneous_tree(root_span, logs.logs)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Dispatch to Agent
        # ═══════════════════════════════════════════════════════════════════════
        if router_output.agent_type == "single_rca":
            return await self.single_rca_agent.chat(
                trace_id=req_data.trace_id,
                chat_id=req_data.chat_id,
                user_message=req_data.message,
                model=req_data.model,
                timestamp=req_data.time,
                tree=tree,
                openai_token=openai_token,
            )
        # elif router_output.agent_type == "code":
        #     return await self.code_agent.chat(...)
        
        # Fallback
        return await self.general_agent.chat(
            trace_id=req_data.trace_id or "",
            chat_id=req_data.chat_id,
            user_message=req_data.message,
            model=req_data.model,
            timestamp=req_data.time,
            openai_token=openai_token,
        )