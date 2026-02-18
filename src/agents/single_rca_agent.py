import asyncio
import json
from datetime import datetime
from openai import AsyncOpenAI
from src.agents.base import BaseAgent
from src.routing.types import ChatbotResponse, ChatModel, MessageType, Reference
from src.context.model import SpanNode
from src.agents.filter.feature import log_feature_selector, span_feature_selector
from src.agents.chunk.sequential import get_trace_context_messages
from src.agents.summarizer.chunk import chunk_summarize
from src.intel.pipeline import IntelligencePipeline
SINGLE_RCA_SYSTEM_PROMPT = """You are TraceRoot, an AI assistant specialized in analyzing distributed traces and logs to find root causes of issues.
Your capabilities:
1. Analyze trace data to identify performance bottlenecks
2. Find root causes of errors and failures
3. Explain the flow of requests through services
4. Identify patterns like N+1 queries, timeouts, and cascading failures
When analyzing traces:
- Focus on spans with errors or high latency
- Look for patterns in log messages
- Consider the timing and order of events
- Cite specific evidence from the trace
Format your response with:
- A clear summary of the issue
- The root cause explanation
- Supporting evidence from the trace
- Recommendations for fixing the issue"""
class SingleRCAAgent(BaseAgent):
    """Root Cause Analysis agent for diagnostic queries."""
    
    def __init__(self):
        self.chat_client = AsyncOpenAI()
        self.system_prompt = SINGLE_RCA_SYSTEM_PROMPT
        self.intel_pipeline = IntelligencePipeline()
    
    async def chat(
        self,
        trace_id: str,
        chat_id: str,
        user_message: str,
        model: ChatModel,
        timestamp: datetime,
        tree: SpanNode,
        chat_history: list[dict] | None = None,
        openai_token: str | None = None,
        **kwargs
    ) -> ChatbotResponse:
        """Main chat entrypoint for RCA."""
        
        client = AsyncOpenAI(api_key=openai_token) if openai_token else self.chat_client
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Run Intelligence Pipeline
        # ═══════════════════════════════════════════════════════════════════════
        intel_result = await self.intel_pipeline.process(
            tree=tree,
            user_query=user_message
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Check Fast Path
        # ═══════════════════════════════════════════════════════════════════════
        if intel_result.fast_path_available:
            # Use pre-computed pattern explanation
            pattern = intel_result.pattern_matches[0]
            response_message = self._format_fast_path_response(
                pattern=pattern,
                ranked_causes=intel_result.ranked_causes,
                user_message=user_message
            )
            
            return ChatbotResponse(
                time=datetime.now(),
                message=response_message,
                reference=self._build_references(intel_result),
                message_type=MessageType.ASSISTANT,
                chat_id=chat_id
            )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Feature Selection (if not using fast path)
        # ═══════════════════════════════════════════════════════════════════════
        log_features, span_features = await asyncio.gather(
            log_feature_selector(user_message, client, model),
            span_feature_selector(user_message, client, model)
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Build Context with Intelligence Enhancement
        # ═══════════════════════════════════════════════════════════════════════
        # Use compressed context from intelligence layer
        context = self._build_enhanced_context(
            compressed_context=intel_result.compressed_context,
            ranked_causes=intel_result.ranked_causes,
            pattern_matches=intel_result.pattern_matches
        )
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Chunk Context
        # ═══════════════════════════════════════════════════════════════════════
        context_chunks = get_trace_context_messages(context)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: LLM Reasoning
        # ═══════════════════════════════════════════════════════════════════════
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add chat history
        if chat_history:
            for h in chat_history[-10:]:
                messages.append({"role": h["role"], "content": h["content"]})
        
        # Process each chunk
        responses = []
        for chunk in context_chunks:
            chunk_messages = messages + [{
                "role": "user",
                "content": f"{chunk}\n\nUser query: {user_message}"
            }]
            
            response = await client.chat.completions.create(
                model=model.value,
                messages=chunk_messages
            )
            responses.append(response.choices[0].message.content)
        
        # ═══════════════════════════════════════════════════════════════════════
        # STEP 7: Summarize if Multiple Chunks
        # ═══════════════════════════════════════════════════════════════════════
        if len(responses) == 1:
            final_response = responses[0]
        else:
            summary = await chunk_summarize(responses, client, model)
            final_response = summary.answer
        
        return ChatbotResponse(
            time=datetime.now(),
            message=final_response,
            reference=self._build_references(intel_result),
            message_type=MessageType.ASSISTANT,
            chat_id=chat_id
        )
    
    def _format_fast_path_response(self, pattern, ranked_causes, user_message):
        """Format response using pattern match (fast path)."""
        return f"""## Root Cause Identified: {pattern.pattern_name}
**Confidence**: {pattern.confidence:.0%}
### Explanation
{pattern.explanation}
### Recommended Fix
{pattern.recommended_fix}
### Evidence
- Pattern category: {pattern.pattern_category}
- Matched spans: {len(pattern.matched_spans)}
- Top ranked cause: {ranked_causes.causes[0].span_function if ranked_causes.causes else 'N/A'}"""
    
    def _build_enhanced_context(self, compressed_context, ranked_causes, pattern_matches):
        """Build context enhanced with intelligence results."""
        intel_prefix = "=== INTELLIGENCE LAYER ANALYSIS ===\n"
        
        if ranked_causes.causes:
            intel_prefix += "\nTop Ranked Causes:\n"
            for cause in ranked_causes.causes[:3]:
                intel_prefix += f"  #{cause.rank} {cause.span_function} (score: {cause.score:.2f})\n"
        
        if pattern_matches:
            intel_prefix += "\nMatched Patterns:\n"
            for pm in pattern_matches[:2]:
                intel_prefix += f"  - {pm.pattern_name} ({pm.confidence:.0%})\n"
        
        intel_prefix += "\n=== TRACE CONTEXT ===\n"
        
        return intel_prefix + json.dumps(compressed_context.compressed_tree, indent=2)
    
    def _build_references(self, intel_result):
        """Build references from intelligence result."""
        refs = []
        for cause in intel_result.ranked_causes.causes[:3]:
            refs.append(Reference(
                type="span",
                span_id=cause.span_id
            ))
        return refs