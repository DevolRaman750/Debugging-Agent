"""
Response Builder — Stage 8: Response to User
==============================================
Assembles the final ChatbotResponse from all pipeline stages.

This is the LAST step before the response reaches the user.
Takes outputs from:
  - Stage 5 (LLM Reasoning) → answer text
  - Stage 6 (Evidence Synthesis) → validated answer + references + confidence
  - Stage 3 (Intelligence Layer) → patterns, ranked causes, timing
  - Stage 7 (Persistence) → stores everything

Produces:
  - ChatbotResponse with metadata containing intelligence info

Usage in SingleRCAAgent:
    from src.routing.response_builder import (
        build_response,
        build_error_response,
        build_general_response,
        response_to_chat_record,
        intel_to_metrics_record,
    )

    response = build_response(
        chat_id=chat_id,
        answer=validated.answer,
        references=validated.references,
        intel_result=intel_result,
        validated=validated,
    )
"""

from datetime import datetime, timezone
from typing import Any, Optional

from src.agents.types import Reference as AgentReference
from src.routing.types import (
    ChatbotResponse,
    IntelligenceMetadata,
    MessageType,
    Reference,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER — build_response()
# ═══════════════════════════════════════════════════════════════════════════════

def build_response(
    chat_id: str,
    answer: str,
    references: list[Reference],
    intel_result: Any,
    validated: Any | None = None,
    fast_path_used: bool = False,
) -> dict:
    """
    Build the final ChatbotResponse dict from all pipeline outputs.

    This is the main function for Stage 8. Call this at the end of
    SingleRCAAgent.chat() to produce the response.

    Args:
        chat_id:         The conversation ID
        answer:          The final answer text (already validated/caveated by synthesizer)
        references:      List of validated references from synthesizer
        intel_result:    IntelligenceResult from the Intelligence Layer
        validated:       ValidatedResponse from synthesizer (Stage 6) — optional
        fast_path_used:  Whether the fast path was used (skipped LLM)

    Returns:
        Dict ready to be passed to ChatbotResponse(**result)
    """

    # ── Build intelligence metadata ──
    metadata = _build_metadata(intel_result, validated, fast_path_used)

    # ── Number the references [1], [2], ... ──
    numbered_refs = _number_references(references)

    # ── Extract validation info ──
    validation_passed = None
    validation_confidence = None
    validation_notes = []
    fallback_used = False

    if validated:
        validation_passed = validated.validation_passed
        validation_confidence = validated.confidence
        validation_notes = validated.validation_notes
        fallback_used = validated.fallback_used
    elif fast_path_used:
        # Fast path skips the synthesizer, but the pattern match itself
        # is high-confidence evidence → treat as validated.
        validation_passed = True
        validation_confidence = 1.0
        validation_notes = ["✅ Fast path — high-confidence pattern match"]

    # ── Assemble the final response ──
    return {
        "time": datetime.now(timezone.utc),
        "message": answer,
        "reference": numbered_refs,
        "message_type": MessageType.ASSISTANT,
        "chat_id": chat_id,
        "action": {
            "type": "agent_chat" if not fast_path_used else "fast_path",
            "status": "success",
        },
        # Validation fields (from your existing ChatbotResponse)
        "validation_passed": validation_passed,
        "validation_confidence": validation_confidence,
        "validation_notes": validation_notes,
        "fallback_used": fallback_used,
        # Stage 8 intelligence metadata
        "metadata": metadata,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def build_error_response(
    chat_id: str,
    error_message: str,
) -> dict:
    """Build a response dict when the pipeline fails.

    Args:
        chat_id:       The conversation ID
        error_message: What went wrong
    """
    return {
        "time": datetime.now(timezone.utc),
        "message": f"I encountered an error while analyzing the trace: {error_message}",
        "reference": [],
        "message_type": MessageType.ASSISTANT,
        "chat_id": chat_id,
        "action": {"type": "agent_chat", "status": "failed"},
        "validation_passed": None,
        "validation_confidence": None,
        "validation_notes": [],
        "fallback_used": False,
        "metadata": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL RESPONSE (no trace context, no metadata)
# ═══════════════════════════════════════════════════════════════════════════════

def build_general_response(
    chat_id: str,
    answer: str,
) -> dict:
    """Build a response dict for general (non-RCA) queries.

    No intelligence metadata since no trace was analyzed.
    """
    return {
        "time": datetime.now(timezone.utc),
        "message": answer,
        "reference": [],
        "message_type": MessageType.ASSISTANT,
        "chat_id": chat_id,
        "action": {"type": "general", "status": "success"},
        "validation_passed": None,
        "validation_confidence": None,
        "validation_notes": [],
        "fallback_used": False,
        "metadata": None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def response_to_chat_record(
    response: ChatbotResponse,
    trace_id: str,
    user_message: str | None = None,
    model: str | None = None,
    mode: str | None = None,
) -> dict:
    """Convert a ChatbotResponse to a record ready for dao.insert_chat_record().

    Call this after build_response() to persist the assistant's message.

    Example:
        resp = ChatbotResponse(**build_response(...))
        record = response_to_chat_record(resp, trace_id="abc123", ...)
        await dao.insert_chat_record(record)
    """
    # Serialize references to plain dicts
    ref_dicts = [
        r.model_dump() if hasattr(r, "model_dump") else r
        for r in response.reference
    ]

    return {
        "chat_id": response.chat_id,
        "timestamp": response.time.isoformat(),
        "role": "assistant",
        "content": response.message,
        "user_content": user_message,
        "trace_id": trace_id,
        "model": model,
        "mode": mode,
        "message_type": "assistant",
        "action_type": "AGENT_CHAT",
        "status": "SUCCESS",
        "user_message": user_message,
        "reference": ref_dicts,
    }


def intel_to_metrics_record(
    intel_result: Any,
    validated: Any | None,
    chat_id: str,
    trace_id: str,
    pipeline_ms: float = 0,
    response: ChatbotResponse | None = None,
) -> dict:
    """Convert Intelligence + Validation results to a record for dao.insert_intelligence_metrics().

    Call this after the pipeline completes to persist metrics for the evaluation loop.

    Args:
        intel_result:  IntelligenceResult from the pipeline
        validated:     ValidatedResponse from synthesizer (None for fast path)
        chat_id:       Conversation ID
        trace_id:      Jaeger trace ID
        pipeline_ms:   Total pipeline wall-clock time (ms)
        response:      ChatbotResponse — used for fast-path validation fallback

    Example:
        metrics = intel_to_metrics_record(intel_result, validated, chat_id, trace_id, pipeline_ms, response)
        await dao.insert_intelligence_metrics(metrics)
    """

    # ── Serialize pattern matches ──
    pattern_list = []
    if hasattr(intel_result, 'pattern_matches') and intel_result.pattern_matches:
        for pm in intel_result.pattern_matches:
            pattern_list.append({
                "name": pm.pattern_name,
                "confidence": pm.confidence,
                "category": pm.pattern_category,
                "explanation": pm.explanation[:200],
                "matched_spans": len(pm.matched_spans),
            })

    # ── Serialize ranked causes (top 5) ──
    cause_list = []
    if hasattr(intel_result, 'ranked_causes') and intel_result.ranked_causes:
        causes = intel_result.ranked_causes.causes if hasattr(intel_result.ranked_causes, 'causes') else []
        for c in causes[:5]:
            cause_list.append({
                "span_function": c.span_function,
                "score": c.score,
                "rank": c.rank,
                "span_id": c.span_id,
            })

    # ── Serialize validation result ──
    validation_dict = None
    if validated:
        validation_dict = {
            "passed": validated.validation_passed,
            "confidence": validated.confidence,
            "issues": len(validated.validation_notes),
            "fallback_used": validated.fallback_used,
        }
    elif response and response.validation_passed is not None:
        # Fast path — validation info lives on the response itself
        validation_dict = {
            "passed": response.validation_passed,
            "confidence": response.validation_confidence,
            "issues": 0,
            "fallback_used": response.fallback_used,
        }

    # ── Compression ratio ──
    compression_ratio = None
    if hasattr(intel_result, 'compressed_context') and intel_result.compressed_context:
        compression_ratio = getattr(intel_result.compressed_context, 'compression_ratio', None)

    return {
        "trace_id": trace_id,
        "chat_id": chat_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pattern_matches": pattern_list,
        "ranked_causes": cause_list,
        "fast_path_used": intel_result.fast_path_available,
        "compression_ratio": compression_ratio,
        "processing_time_ms": pipeline_ms,
        "validation_result": validation_dict,
        "user_feedback": None,  # Filled later when user clicks 👍/👎
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_metadata(
    intel_result: Any,
    validated: Any | None,
    fast_path_used: bool,
) -> IntelligenceMetadata:
    """Build IntelligenceMetadata from pipeline outputs.

    Determines confidence level based on:
      - Top cause score from ranked causes
      - Validation results from evidence synthesizer
      - Whether fast path was used
    """

    # ── Determine confidence ──
    confidence = _determine_confidence(intel_result, validated, fast_path_used)

    # ── Pattern info ──
    pattern_name = None
    if hasattr(intel_result, 'pattern_matches') and intel_result.pattern_matches:
        pattern_name = intel_result.pattern_matches[0].pattern_name

    # ── Top cause info ──
    top_cause = None
    top_cause_score = None
    causes_found = 0

    if hasattr(intel_result, 'ranked_causes') and intel_result.ranked_causes:
        causes = intel_result.ranked_causes.causes if hasattr(intel_result.ranked_causes, 'causes') else []
        causes_found = len(causes)
        if causes:
            top_cause = causes[0].span_function
            top_cause_score = causes[0].score

    return IntelligenceMetadata(
        confidence=confidence,
        processing_time_ms=getattr(intel_result, 'processing_time_ms', 0),
        pattern_matched=pattern_name,
        fast_path=fast_path_used,
        top_cause=top_cause,
        top_cause_score=top_cause_score,
        causes_found=causes_found,
    )


def _determine_confidence(
    intel_result: Any,
    validated: Any | None,
    fast_path_used: bool,
) -> str:
    """Determine the confidence label: HIGH, MEDIUM, or LOW.

    Logic from kan.md:
      HIGH:   fast_path used OR top_cause_score > 0.8 with clear lead
      MEDIUM: top_cause_score > 0.6
      LOW:    top_cause_score < 0.6 OR validation failed badly
    """

    # Fast path = high confidence pattern match → always HIGH
    if fast_path_used:
        return "HIGH"

    # Validation fallback = response was unreliable → LOW
    if validated and validated.fallback_used:
        return "LOW"

    # Check ranked causes
    ranked = getattr(intel_result, 'ranked_causes', None)
    if ranked:
        causes = ranked.causes if hasattr(ranked, 'causes') else []
        if causes:
            top_score = causes[0].score

            # Check gap between #1 and #2 cause
            if len(causes) >= 2:
                gap = top_score - causes[1].score
            else:
                gap = top_score

            if top_score > 0.8 and gap > 0.2:
                return "HIGH"
            elif top_score > 0.6:
                return "MEDIUM"
            else:
                return "LOW"

    return "LOW"


def _number_references(references: list) -> list[Reference]:
    """Assign sequential numbers [1], [2], ... to references.

    Creates new Reference objects with the number field set.
    """
    numbered = []
    for i, ref in enumerate(references):
        if isinstance(ref, (Reference, AgentReference)):
            ref_dict = ref.model_dump()
        elif isinstance(ref, dict):
            ref_dict = ref
        else:
            ref_dict = {"type": "unknown"}

        ref_dict["number"] = i + 1
        # Only keep fields that Reference accepts
        valid = {k: v for k, v in ref_dict.items() if k in Reference.model_fields}
        numbered.append(Reference(**valid))

    return numbered
