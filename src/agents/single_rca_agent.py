"""
SingleRCAAgent - Root Cause Analysis agent for diagnostic queries.

This agent is the heart of TraceRoot's RCA capability. It takes a span tree
and a user question, runs the Intelligence Layer, and produces a human-readable
root cause explanation.

TWO PATHS:
  Fast Path: Intelligence Layer found a known pattern → skip LLM, return pattern explanation
  Full Path: No high-confidence pattern → build context → send to LLM for reasoning
"""

import asyncio
import json
import time as _timer
from datetime import datetime, timezone
from typing import Any
from openai import AsyncOpenAI

from src.agents.base import BaseAgent
from src.routing.types import ChatbotResponse, ChatModel, MessageType, Reference
from src.context.model import SpanNode
from src.agents.filter.feature import log_feature_selector, span_feature_selector
from src.agents.chunk.sequential import get_trace_context_messages
from src.agents.summarizer.chunk import chunk_summarize
from src.agents.types import LogFeature, SpanFeature, ChatOutput
from src.intel.pipeline import IntelligencePipeline
from src.intel.synthesizer import validate_response
from src.routing.response_builder import (
    build_response,
    response_to_chat_record,
    intel_to_metrics_record,
)
from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL


# ═══════════════════════════════════════════════════════════════════════════════
# RCA SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════
SINGLE_RCA_SYSTEM_PROMPT = """You are TraceRoot, an AI assistant specialized in analyzing distributed traces and logs to find root causes of issues.

Your capabilities:
1. Analyze trace data to identify performance bottlenecks
2. Find root causes of errors and failures
3. Explain the flow of requests through services
4. Identify patterns like N+1 queries, timeouts, and cascading failures

You will receive:
- INTELLIGENCE INSIGHTS: Pre-analyzed data showing ranked causes, detected patterns, and failure reports
- TRACE CONTEXT: The actual span tree with selected fields relevant to the user's question

When analyzing traces:
- Focus on spans with errors or high latency
- Look for patterns in log messages
- Consider the timing and order of events
- Cite specific evidence from the trace using [1], [2], etc.

Format your response with:
- A clear summary of the issue
- The root cause explanation with evidence
- Recommendations for fixing the issue"""


class SingleRCAAgent(BaseAgent):
    """Root Cause Analysis agent for diagnostic queries."""

    def __init__(self, groq_api_key: str = None, client: AsyncOpenAI = None):
        api_key = groq_api_key or GROQ_API_KEY
        if client:
            self.chat_client = client
        elif api_key:
            self.chat_client = AsyncOpenAI(
                api_key=api_key,
                base_url=GROQ_BASE_URL
            )
        else:
            self.chat_client = None

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
        groq_api_key: str | None = None,
        db_client=None,
        **kwargs
    ) -> ChatbotResponse:
        """Main chat entrypoint for RCA."""

        # Use provided key or default client
        if groq_api_key:
            client = AsyncOpenAI(api_key=groq_api_key, base_url=GROQ_BASE_URL)
        elif self.chat_client:
            client = self.chat_client
        else:
            return ChatbotResponse(
                time=datetime.now(),
                message="Error: No LLM client configured. Provide a Groq API key.",
                reference=[],
                message_type=MessageType.ASSISTANT,
                chat_id=chat_id,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: Run Intelligence Pipeline
        # ═══════════════════════════════════════════════════════════════════════
        _t0 = _timer.perf_counter()
        intel_result = await self.intel_pipeline.process(
            tree=tree,
            user_query=user_message
        )
        _pipeline_ms = (_timer.perf_counter() - _t0) * 1000

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Check Fast Path
        # ═══════════════════════════════════════════════════════════════════════
        if intel_result.fast_path_available:
            # Use pre-computed pattern explanation — NO LLM call needed!
            pattern = intel_result.pattern_matches[0]
            response_message = self._format_fast_path_response(
                pattern=pattern,
                ranked_causes=intel_result.ranked_causes,
                user_message=user_message,
            )

            fast_response = ChatbotResponse(**build_response(
                chat_id=chat_id,
                answer=response_message,
                references=self._build_references(intel_result),
                intel_result=intel_result,
                validated=None,
                fast_path_used=True,
            ))

            # ── Persist (fast path) ──
            if db_client:
                await self._persist(
                    db_client=db_client,
                    trace_id=trace_id,
                    chat_id=chat_id,
                    user_message=user_message,
                    response=fast_response,
                    intel_result=intel_result,
                    pipeline_ms=_pipeline_ms,
                    validated=None,
                )

            return fast_response

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Feature Selection (asks LLM which fields matter)
        #
        # User asks "why is this slow?" → LLM picks: [SPAN_LATENCY]
        # User asks "show me errors"    → LLM picks: [LOG_LEVEL, LOG_MESSAGE_VALUE]
        # ═══════════════════════════════════════════════════════════════════════
        log_features, span_features = await asyncio.gather(
            log_feature_selector(user_message, client, GROQ_MODEL),
            span_feature_selector(user_message, client, GROQ_MODEL),
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: Build Context with Intelligence Enhancement
        #
        # This is where _build_context and _serialize_tree come in.
        # They convert the compressed dict into a human-readable text
        # with ONLY the selected fields.
        # ═══════════════════════════════════════════════════════════════════════
        context = self._build_context(
            intel_result=intel_result,
            log_features=log_features,
            span_features=span_features,
        )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Chunk Context (split if > 200k chars)
        # ═══════════════════════════════════════════════════════════════════════
        context_chunks = get_trace_context_messages(context)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: LLM Reasoning (send each chunk to Groq)
        # ═══════════════════════════════════════════════════════════════════════
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add chat history for multi-turn context
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
                model=GROQ_MODEL,
                messages=chunk_messages,
                temperature=0.5,
            )
            responses.append(response.choices[0].message.content)

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 7: Summarize if Multiple Chunks
        # ═══════════════════════════════════════════════════════════════════════
        if len(responses) == 1:
            final_response = responses[0]
        else:
            summary = await chunk_summarize(
                response_answers=responses,
                client=client,
                model=GROQ_MODEL,
            )
            final_response = summary.answer

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 8: Evidence Synthesis & Validation
        #
        # Validates the LLM's answer against actual trace data.
        # Catches hallucinations: invalid span refs, unsupported claims,
        # inconsistencies with Intelligence Layer findings.
        #
        # Three outcomes:
        #   PASS     → answer returned as-is
        #   CAVEAT   → warning notes appended to answer
        #   FALLBACK → answer replaced with pattern-based explanation
        # ═══════════════════════════════════════════════════════════════════════
        raw_refs = self._build_references(intel_result)
        chat_output = ChatOutput(
            answer=final_response,
            reference=[r.model_dump() for r in raw_refs],
        )

        validated = validate_response(
            chat_output=chat_output,
            intel_result=intel_result,
            tree=intel_result.classified_tree,
        )

        llm_response = ChatbotResponse(**build_response(
            chat_id=chat_id,
            answer=validated.answer,
            references=validated.references,
            intel_result=intel_result,
            validated=validated,
            fast_path_used=False,
        ))

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 9: Persist to Database
        #
        # Stores chat record, chat metadata, and intelligence metrics.
        # Runs only when a db_client is provided (None in unit tests).
        # ═══════════════════════════════════════════════════════════════════════
        if db_client:
            await self._persist(
                db_client=db_client,
                trace_id=trace_id,
                chat_id=chat_id,
                user_message=user_message,
                response=llm_response,
                intel_result=intel_result,
                pipeline_ms=_pipeline_ms,
                validated=validated,
            )

        return llm_response

    # ═══════════════════════════════════════════════════════════════════════════
    # PERSISTENCE — Writes records to the database (SQLite or MongoDB)
    # ═══════════════════════════════════════════════════════════════════════════

    async def _persist(
        self,
        db_client,
        trace_id: str,
        chat_id: str,
        user_message: str,
        response: ChatbotResponse,
        intel_result,
        pipeline_ms: float,
        validated=None,
    ):
        """Persist chat record, chat metadata, and intelligence metrics.

        Called after both fast-path and LLM-path responses.
        Uses centralised helpers from response_builder.py.
        Failures are logged but never crash the response.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        try:
            # ── 1. User message record ──
            await db_client.insert_chat_record({
                "chat_id": chat_id,
                "timestamp": now_iso,
                "role": "user",
                "content": user_message,
                "trace_id": trace_id,
                "message_type": "user",
                "action_type": "AGENT_CHAT",
                "status": "SUCCESS",
            })

            # ── 2. Assistant message record (via response_to_chat_record) ──
            assistant_record = response_to_chat_record(
                response=response,
                trace_id=trace_id,
                user_message=user_message,
                model=GROQ_MODEL,
                mode="AGENT",
            )
            await db_client.insert_chat_record(assistant_record)

            # ── 3. Chat metadata (upsert) ──
            title = f"RCA: {user_message[:50]}"
            await db_client.insert_chat_metadata({
                "chat_id": chat_id,
                "timestamp": now_iso,
                "chat_title": title,
                "trace_id": trace_id,
            })

            # ── 4. Intelligence metrics (via intel_to_metrics_record) ──
            metrics_record = intel_to_metrics_record(
                intel_result=intel_result,
                validated=validated,
                chat_id=chat_id,
                trace_id=trace_id,
                pipeline_ms=pipeline_ms,
                response=response,
            )
            await db_client.insert_intelligence_metrics(metrics_record)

        except Exception as e:
            # Never crash the response because of a DB write failure
            print(f"  ⚠️  DB persist error (non-fatal): {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CONTEXT BUILDING — Converts Intelligence Layer output into LLM-ready text
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_context(
        self,
        intel_result,
        log_features: list[LogFeature],
        span_features: list[SpanFeature],
    ) -> str:
        """Build the final context string that gets sent to the LLM.

        This is the KEY method that bridges Intelligence Layer → LLM.

        It combines three things:
          1. Intelligence insights (ranked causes, patterns, failures)
          2. Serialized trace tree (filtered to only selected features)
          3. Suppression info (what was removed)

        EXAMPLE OUTPUT:
            === INTELLIGENCE INSIGHTS ===
            Top Ranked Causes:
              #1 product_db_query (score: 0.95)
            Matched Patterns:
              - Slow Query (90%)
            Failures Detected: 1
              - product_db_query: timeout

            === TRACE CONTEXT ===
            POST /api/checkout (3200ms)
            ├── auth_service (50ms)
            ├── product_db_query (2500ms) ⚠️
            │   [WARNING] Query execution plan: FULL TABLE SCAN
            │   [INFO] Query returned 47 rows
            └── payment_gateway (200ms)

            [Suppressed: 3 noise spans removed]
        """
        parts = []

        # ──────────────────────────────────────────────────────────────────────
        # PART 1: Intelligence Layer Insights (ranked causes + patterns)
        # ──────────────────────────────────────────────────────────────────────
        parts.append("=== INTELLIGENCE INSIGHTS ===")

        # Ranked causes
        ranked = intel_result.ranked_causes
        if ranked.causes:
            parts.append(f"\nConfidence: {ranked.confidence_level.value.upper()}")
            parts.append("Top Ranked Causes:")
            for cause in ranked.causes[:3]:
                factors_str = ", ".join(
                    f"{f.factor_name}={f.factor_value:.2f}"
                    for f in cause.contributing_factors[:3]
                )
                parts.append(
                    f"  #{cause.rank} {cause.span_function} "
                    f"(score: {cause.score:.2f}) [{factors_str}]"
                )

        # Pattern matches
        if intel_result.pattern_matches:
            parts.append("\nMatched Patterns:")
            for pm in intel_result.pattern_matches:
                parts.append(f"  - {pm.pattern_name} ({pm.confidence:.0%})")
                parts.append(f"    Explanation: {pm.explanation}")
                parts.append(f"    Recommended Fix: {pm.recommended_fix}")

        # Failure report
        failures = intel_result.failure_report
        if failures.has_failures:
            parts.append(f"\nFailures Detected: {len(failures.failure_spans)}")
            for fs in failures.failure_spans:
                root_cause_marker = " ← ROOT CAUSE" if fs.is_root_cause else ""
                parts.append(
                    f"  - {fs.span_function}: {fs.failure_type.value}"
                    f"{root_cause_marker}"
                )
                if fs.error_messages:
                    # Show first error message (truncated)
                    parts.append(f"    Error: {fs.error_messages[0][:100]}")

            # Show failure chains (propagation paths)
            if failures.failure_chains:
                parts.append("\nFailure Propagation:")
                for chain in failures.failure_chains:
                    parts.append(f"  {chain.propagation_path}")

        # ──────────────────────────────────────────────────────────────────────
        # PART 2: Serialized Trace Tree (with only selected features)
        # ──────────────────────────────────────────────────────────────────────
        parts.append("\n\n=== TRACE CONTEXT ===")

        # compressed_tree is a dict from the Intelligence Layer's compressor
        compressed_tree = intel_result.compressed_context.compressed_tree
        tree_text = self._serialize_tree(
            node=compressed_tree,
            log_features=log_features,
            span_features=span_features,
            depth=0,
        )
        parts.append(tree_text)

        # ──────────────────────────────────────────────────────────────────────
        # PART 3: Suppression Stats
        # ──────────────────────────────────────────────────────────────────────
        stats = intel_result.suppression_stats
        if stats.suppressed_count > 0:
            parts.append(
                f"\n[Suppressed: {stats.suppressed_count} noise spans removed, "
                f"compression ratio: {stats.compression_ratio:.1f}x]"
            )

        # Compression stats
        ctx = intel_result.compressed_context
        parts.append(
            f"[Context: {ctx.original_token_count} tokens → "
            f"{ctx.token_count} tokens ({ctx.compression_ratio:.1f}x compression)]"
        )

        return "\n".join(parts)

    def _serialize_tree(
        self,
        node: dict,
        log_features: list[LogFeature],
        span_features: list[SpanFeature],
        depth: int = 0,
        is_last: bool = True,
    ) -> str:
        """Convert compressed tree dict into human-readable text for the LLM.

        Only includes fields selected by the Feature Selector.

        Args:
            node: A dict from compressed_context.compressed_tree
                  Format: {"function": "...", "latency_ms": ..., "type": "...",
                           "logs": [...], "log_summary": "...", "children": [...]}
            log_features: Which log fields to display
            span_features: Which span fields to display
            depth: Current tree depth (for indentation)
            is_last: Whether this is the last child (for tree drawing)

        Returns:
            Human-readable tree text like:
                POST /api/checkout (3200ms) [business_logic]
                ├── auth_service (50ms) [http_handler]
                ├── product_db_query (2500ms) [db_query] ⚠️
                │   [WARNING] Full table scan detected
                └── payment_gateway (200ms) [http_request]
        """

        # ── Tree-drawing prefixes ──
        if depth == 0:
            prefix = ""
            child_prefix = ""
        else:
            prefix = ("└── " if is_last else "├── ")
            child_prefix = ("    " if is_last else "│   ")

        # Add indentation for depth
        indent = ""
        if depth > 1:
            indent = "│   " * (depth - 1)
        elif depth == 1:
            indent = ""

        full_prefix = indent + prefix

        # ── Span line: function name + selected features ──
        func_name = node.get("function", "unknown")
        span_line = f"{full_prefix}{func_name}"

        # Add latency if selected
        if SpanFeature.SPAN_LATENCY in span_features:
            latency = node.get("latency_ms", 0)
            span_line += f" ({latency:.0f}ms)"

        # Add span type as tag
        span_type = node.get("type", "unknown")
        span_line += f" [{span_type}]"

        # Add warning/error markers
        logs = node.get("logs", [])
        log_summary = node.get("log_summary", "")
        has_errors = any(
            l.get("level", "").upper() in ("ERROR", "CRITICAL") for l in logs
        )
        if has_errors:
            span_line += " ❌"
        elif "error" in log_summary.lower():
            span_line += " ⚠️"

        # Add timestamps if selected
        if SpanFeature.SPAN_UTC_START_TIME in span_features:
            start = node.get("start_time", "")
            if start:
                span_line += f" @{start}"

        lines = [span_line]

        # ── Log lines: only show selected log features ──
        log_indent = indent + child_prefix

        if logs:
            # Full logs available (top cause spans)
            for log_entry in logs:
                log_parts = []

                if LogFeature.LOG_LEVEL in log_features:
                    level = log_entry.get("level", "INFO")
                    log_parts.append(f"[{level}]")

                if LogFeature.LOG_MESSAGE_VALUE in log_features:
                    msg = log_entry.get("message", "")
                    log_parts.append(msg[:150])  # Truncate long messages

                if LogFeature.LOG_FUNC_NAME in log_features:
                    func = log_entry.get("func_name", "")
                    if func:
                        log_parts.append(f"in {func}")

                if LogFeature.LOG_FILE_NAME in log_features:
                    fname = log_entry.get("file_name", "")
                    if fname:
                        log_parts.append(f"({fname})")

                if LogFeature.LOG_LINE_NUMBER in log_features:
                    line_no = log_entry.get("line_number", "")
                    if line_no:
                        log_parts.append(f"L{line_no}")

                if log_parts:
                    lines.append(f"{log_indent}  {' '.join(log_parts)}")

        elif log_summary:
            # Summarized logs (non-top-cause spans)
            lines.append(f"{log_indent}  ({log_summary})")

        # ── Recurse into children ──
        children = node.get("children", [])
        for i, child in enumerate(children):
            is_child_last = (i == len(children) - 1)
            child_text = self._serialize_tree(
                node=child,
                log_features=log_features,
                span_features=span_features,
                depth=depth + 1,
                is_last=is_child_last,
            )
            lines.append(child_text)

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════

    def _format_fast_path_response(self, pattern, ranked_causes, user_message):
        """Format response using pattern match (fast path — no LLM needed)."""
        top_cause = (
            ranked_causes.causes[0].span_function
            if ranked_causes.causes else "N/A"
        )

        return f"""## 🔍 Root Cause Identified: {pattern.pattern_name}

**Confidence**: {pattern.confidence:.0%}

### Explanation
{pattern.explanation}

### Recommended Fix
{pattern.recommended_fix}

### Evidence
- **Pattern category**: {pattern.pattern_category}
- **Matched spans**: {len(pattern.matched_spans)}
- **Top ranked cause**: {top_cause}
- **Evidence**: {', '.join(pattern.matched_evidence[:3])}"""

    def _build_references(self, intel_result):
        """Build references from intelligence result."""
        refs = []

        # Add top ranked causes as references
        for cause in intel_result.ranked_causes.causes[:3]:
            refs.append(Reference(
                type="span",
                span_id=cause.span_id,
                span_function_name=cause.span_function,
            ))

        # Add pattern-matched spans
        for pm in intel_result.pattern_matches[:2]:
            for span_id in pm.matched_spans[:2]:
                if not any(r.span_id == span_id for r in refs):
                    refs.append(Reference(
                        type="span",
                        span_id=span_id,
                    ))

        return refs
