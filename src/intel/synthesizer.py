"""
Evidence Synthesis & Validation
================================
Stage 6 of the TraceRoot pipeline.

Takes the LLM's raw answer + references, validates them against the actual
trace data, and produces a ValidatedResponse. Catches hallucinations before
they reach the user.

THREE CHECKS:
  1. Reference Verification — do cited span_ids exist in the tree?
  2. Claim Verification — does "timeout" claim match actual data?
  3. Consistency Check — does LLM agree with Intelligence Layer?

THREE OUTCOMES:
  PASS     → return answer as-is
  CAVEAT   → add warning notes to the answer
  FALLBACK → replace answer with pattern explanation

NO LLM CALLS — this is pure Python validation logic.
"""

import re
from src.intel.types import (
    ClassifiedSpanNode,
    IntelligenceResult,
    FailureReport,
    PatternMatch,
    RankedCauses,
    SpanType,
    ValidationIssue,
    ValidatedResponse,
)
from src.agents.types import ChatOutput
from src.routing.types import Reference


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def validate_response(
    chat_output: ChatOutput,
    intel_result: IntelligenceResult,
    tree: ClassifiedSpanNode,
) -> ValidatedResponse:
    """
    Main entry point for evidence synthesis.

    Takes the LLM's raw output and validates it against the actual trace data
    and intelligence analysis.

    Args:
        chat_output: The LLM's raw answer + references
        intel_result: The Intelligence Layer's analysis (ranked causes, patterns, etc.)
        tree: The classified span tree (source of truth)

    Returns:
        ValidatedResponse with validated answer, cleaned references, and confidence score
    """
    all_issues: list[ValidationIssue] = []

    # ── Build span lookup for fast access ──
    span_lookup = _build_span_lookup(tree)
    all_spans = list(span_lookup.values())

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 1: Reference Verification
    # ═══════════════════════════════════════════════════════════════════════════
    valid_refs, ref_issues = _verify_references(
        references=chat_output.reference,
        span_lookup=span_lookup,
        tree=tree,
    )
    all_issues.extend(ref_issues)

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 2: Claim Verification
    # ═══════════════════════════════════════════════════════════════════════════
    claim_issues = _verify_claims(
        answer=chat_output.answer,
        tree=tree,
        all_spans=all_spans,
    )
    all_issues.extend(claim_issues)

    # ═══════════════════════════════════════════════════════════════════════════
    # CHECK 3: Consistency Check
    # ═══════════════════════════════════════════════════════════════════════════
    consistency_issues = _check_consistency(
        answer=chat_output.answer,
        intel_result=intel_result,
    )
    all_issues.extend(consistency_issues)

    # ═══════════════════════════════════════════════════════════════════════════
    # DECISION: Pass / Caveat / Fallback
    # ═══════════════════════════════════════════════════════════════════════════
    action, confidence = _decide_action(all_issues)

    if action == "fallback":
        # Major issues — fall back to pattern-based explanation
        fallback_answer = _build_fallback_answer(intel_result)
        return ValidatedResponse(
            answer=fallback_answer,
            references=valid_refs,
            confidence=confidence,
            validation_passed=False,
            issues=all_issues,
            validation_notes=[
                "⚠️ LLM response had significant issues. Using pattern-based explanation instead.",
                f"Issues found: {len(all_issues)} ({_count_by_severity(all_issues)})",
            ],
            fallback_used=True,
        )

    elif action == "caveat":
        # Minor issues — add caveats to the answer
        caveated_answer = _add_caveats(chat_output.answer, all_issues)
        return ValidatedResponse(
            answer=caveated_answer,
            references=valid_refs,
            confidence=confidence,
            validation_passed=True,
            issues=all_issues,
            validation_notes=[
                f"Response validated with {len(all_issues)} minor issue(s).",
            ],
            fallback_used=False,
        )

    else:
        # Clean pass — no issues
        return ValidatedResponse(
            answer=chat_output.answer,
            references=valid_refs,
            confidence=confidence,
            validation_passed=True,
            issues=all_issues,
            validation_notes=["✅ All validation checks passed."],
            fallback_used=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 1: REFERENCE VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_span_lookup(tree: ClassifiedSpanNode) -> dict[str, ClassifiedSpanNode]:
    """Recursively build a flat dict of span_id → span for fast lookups."""
    lookup = {}

    def traverse(span: ClassifiedSpanNode):
        lookup[span.span_id] = span
        for child in span.children_spans:
            traverse(child)

    traverse(tree)
    return lookup


def _verify_references(
    references: list[Reference],
    span_lookup: dict[str, ClassifiedSpanNode],
    tree: ClassifiedSpanNode,
) -> tuple[list[Reference], list[ValidationIssue]]:
    """
    Verify every reference the LLM cited actually exists in the trace tree.

    For each reference:
    - type="span" → Check span_id exists in span_lookup
    - type="log"  → Check log_message exists in the referenced span's logs
    - type="code" → Keep as-is (can't verify source code from trace)

    Returns:
        (valid_references, issues_found)
    """
    valid_refs = []
    issues = []

    for ref in references:
        if ref.type == "span":
            if ref.span_id and ref.span_id in span_lookup:
                # Span exists — valid reference
                valid_refs.append(ref)
            elif ref.span_id:
                # Span NOT found in tree — invalid reference
                issues.append(ValidationIssue(
                    issue_type="invalid_ref",
                    severity="warning",
                    description=f"Referenced span_id '{ref.span_id}' not found in trace tree",
                    span_id=ref.span_id,
                ))
            else:
                # No span_id provided
                issues.append(ValidationIssue(
                    issue_type="invalid_ref",
                    severity="info",
                    description="Reference of type 'span' has no span_id",
                ))

        elif ref.type == "log":
            if ref.span_id and ref.span_id in span_lookup:
                span = span_lookup[ref.span_id]
                # Check if the log message exists in this span's logs
                log_found = _find_log_in_span(span, ref.log_message)
                if log_found:
                    valid_refs.append(ref)
                else:
                    issues.append(ValidationIssue(
                        issue_type="invalid_ref",
                        severity="warning",
                        description=(
                            f"Log message not found in span '{ref.span_id}': "
                            f"'{ref.log_message[:50]}...'" if ref.log_message else "No log message"
                        ),
                        span_id=ref.span_id,
                        original_text=ref.log_message,
                    ))
            elif ref.log_message:
                # No span_id — search all spans for the log message
                found_in_span = _find_log_globally(tree, ref.log_message)
                if found_in_span:
                    # Found it somewhere — keep the reference but note the fix
                    ref.span_id = found_in_span
                    valid_refs.append(ref)
                    issues.append(ValidationIssue(
                        issue_type="invalid_ref",
                        severity="info",
                        description=f"Log reference had no span_id; found in span '{found_in_span}'",
                        span_id=found_in_span,
                    ))
                else:
                    issues.append(ValidationIssue(
                        issue_type="invalid_ref",
                        severity="warning",
                        description=f"Log message not found anywhere in trace: '{ref.log_message[:50]}'",
                        original_text=ref.log_message,
                    ))
            else:
                valid_refs.append(ref)  # Keep generic log refs

        else:
            # "code" or unknown type — keep as-is
            valid_refs.append(ref)

    return valid_refs, issues


def _find_log_in_span(span: ClassifiedSpanNode, log_message: str | None) -> bool:
    """Check if a log message exists in a specific span's logs."""
    if not log_message:
        return False

    target = log_message.lower().strip()

    for log in span.logs:
        msg = str(getattr(log, 'log_message', log)).lower().strip()
        # Fuzzy match: check if target is contained in the log or vice versa
        if target in msg or msg in target:
            return True

    return False


def _find_log_globally(tree: ClassifiedSpanNode, log_message: str) -> str | None:
    """Search entire tree for a log message. Returns span_id if found."""
    if _find_log_in_span(tree, log_message):
        return tree.span_id

    for child in tree.children_spans:
        result = _find_log_globally(child, log_message)
        if result:
            return result

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 2: CLAIM VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

# Each claim type has keywords to detect and a verify function
CLAIM_VERIFIERS = {
    "timeout": {
        "keywords": [r"timeout", r"timed?\s*out", r"deadline\s+exceeded"],
        "description": "Timeout claim",
    },
    "n_plus_1": {
        "keywords": [r"n\+1", r"n\s*plus\s*1", r"repeated\s+quer", r"individual\s+quer"],
        "description": "N+1 query claim",
    },
    "slow_query": {
        "keywords": [r"slow\s+query", r"slow\s+database", r"long\s+query", r"full\s+table\s+scan"],
        "description": "Slow query claim",
    },
    "error_failure": {
        "keywords": [r"error\s+occurred", r"failed\s+with", r"exception\s+thrown", r"500\s+internal"],
        "description": "Error/failure claim",
    },
    "retry": {
        "keywords": [r"retr(?:y|ied)", r"multiple\s+attempt", r"backoff"],
        "description": "Retry pattern claim",
    },
    "connection_refused": {
        "keywords": [r"connection\s+refused", r"connection\s+reset", r"ECONNREFUSED"],
        "description": "Connection refused claim",
    },
    "rate_limit": {
        "keywords": [r"rate\s+limit", r"429", r"too\s+many\s+requests", r"throttl"],
        "description": "Rate limiting claim",
    },
}


def _verify_claims(
    answer: str,
    tree: ClassifiedSpanNode,
    all_spans: list[ClassifiedSpanNode],
) -> list[ValidationIssue]:
    """
    Extract factual claims from the LLM's answer and verify them
    against the actual trace data.

    Scans for keywords like "timeout", "N+1", "error" and checks
    if the trace data actually contains evidence for those claims.
    """
    issues = []
    answer_lower = answer.lower()

    for claim_type, config in CLAIM_VERIFIERS.items():
        # Check if the answer mentions this claim type
        claim_found = False
        matched_keyword = None
        for pattern in config["keywords"]:
            match = re.search(pattern, answer_lower)
            if match:
                claim_found = True
                matched_keyword = match.group()
                break

        if not claim_found:
            continue

        # Claim was made — verify against trace data
        verified = _verify_single_claim(claim_type, tree, all_spans)

        if not verified:
            issues.append(ValidationIssue(
                issue_type="unsupported_claim",
                severity="warning",
                description=(
                    f"{config['description']}: LLM mentions '{matched_keyword}' "
                    f"but no supporting evidence found in the trace data"
                ),
                original_text=matched_keyword,
            ))

    return issues


def _verify_single_claim(
    claim_type: str,
    tree: ClassifiedSpanNode,
    all_spans: list[ClassifiedSpanNode],
) -> bool:
    """Verify a specific claim type against the trace data."""

    if claim_type == "timeout":
        # Check: any span has very high latency OR logs mention timeout
        for span in all_spans:
            if span.span_latency > 1000:  # > 1 second
                return True
            for log in span.logs:
                msg = str(getattr(log, 'log_message', log)).lower()
                if any(kw in msg for kw in ["timeout", "timed out", "deadline"]):
                    return True
        return False

    elif claim_type == "n_plus_1":
        # Check: multiple similar spans under the same parent
        return _has_repeated_queries(tree)

    elif claim_type == "slow_query":
        # Check: any DB-type span with latency > 1s
        for span in all_spans:
            if span.span_type == SpanType.DB_QUERY and span.span_latency > 1000:
                return True
            # Also check by function name pattern
            func_lower = span.func_full_name.lower()
            if any(kw in func_lower for kw in ["query", "select", "insert", "db_"]):
                if span.span_latency > 1000:
                    return True
        return False

    elif claim_type == "error_failure":
        # Check: any span has ERROR or CRITICAL logs
        for span in all_spans:
            for log in span.logs:
                level = str(getattr(log, 'log_level', '')).upper()
                if level in ('ERROR', 'CRITICAL'):
                    return True
        return False

    elif claim_type == "retry":
        # Check: spans with "retry" or "attempt" in name, or repeated similar spans
        for span in all_spans:
            func_lower = span.func_full_name.lower()
            if any(kw in func_lower for kw in ["retry", "attempt", "backoff"]):
                return True
        return _has_repeated_queries(tree)  # Repeated calls also suggest retries

    elif claim_type == "connection_refused":
        # Check: logs mention connection errors
        for span in all_spans:
            for log in span.logs:
                msg = str(getattr(log, 'log_message', log)).lower()
                if any(kw in msg for kw in ["connection refused", "connection reset", "econnrefused"]):
                    return True
        return False

    elif claim_type == "rate_limit":
        # Check: logs mention rate limiting or 429
        for span in all_spans:
            for log in span.logs:
                msg = str(getattr(log, 'log_message', log)).lower()
                if any(kw in msg for kw in ["rate limit", "429", "too many requests", "throttl"]):
                    return True
        return False

    # Unknown claim type — assume valid
    return True


def _has_repeated_queries(tree: ClassifiedSpanNode) -> bool:
    """Check if the tree has an N+1 pattern (3+ similar children under same parent)."""
    # Check immediate children for similar function names
    if len(tree.children_spans) >= 3:
        func_counts: dict[str, int] = {}
        for child in tree.children_spans:
            # Normalize function name: strip trailing numbers/indices
            base_name = re.sub(r'_?\d+$', '', child.func_full_name.lower())
            func_counts[base_name] = func_counts.get(base_name, 0) + 1

        if any(count >= 3 for count in func_counts.values()):
            return True

    # Recurse into children
    for child in tree.children_spans:
        if _has_repeated_queries(child):
            return True

    return False


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK 3: CONSISTENCY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def _check_consistency(
    answer: str,
    intel_result: IntelligenceResult,
) -> list[ValidationIssue]:
    """
    Verify the LLM's conclusion aligns with the Intelligence Layer's analysis.

    Checks:
    1. Does the answer mention the top-ranked cause?
    2. Does the answer align with detected patterns?
    3. Does the answer contradict the failure report?
    """
    issues = []
    answer_lower = answer.lower()

    # ── Check 1: Top ranked cause mentioned? ──
    ranked = intel_result.ranked_causes
    if ranked.causes and ranked.top_cause_score > 0.5:
        top_cause = ranked.causes[0]
        top_func = top_cause.span_function.lower()

        # Extract meaningful parts of the function name
        # e.g., "product_db_query" → check for "product", "db", "query"
        func_parts = re.split(r'[_\-./\s]+', top_func)
        meaningful_parts = [p for p in func_parts if len(p) > 2]

        # Check if any part of the function name appears in the answer
        parts_found = sum(1 for part in meaningful_parts if part in answer_lower)

        if parts_found == 0 and ranked.top_cause_score > 0.7:
            issues.append(ValidationIssue(
                issue_type="inconsistency",
                severity="warning",
                description=(
                    f"Top ranked cause '{top_cause.span_function}' "
                    f"(score: {top_cause.score:.2f}) is not mentioned in the LLM response. "
                    f"The Intelligence Layer identified this as the primary cause."
                ),
                span_id=top_cause.span_id,
            ))

    # ── Check 2: Pattern match alignment ──
    if intel_result.pattern_matches:
        top_pattern = intel_result.pattern_matches[0]

        # Check if the pattern type is reflected in the answer
        pattern_keywords = {
            "n_plus_1_query": ["n+1", "n plus 1", "repeated quer", "batch", "individual quer"],
            "slow_query": ["slow", "latency", "performance", "long", "table scan", "index"],
            "retry": ["retry", "retried", "attempt", "backoff"],
            "timeout": ["timeout", "timed out", "deadline"],
            "connection_pool": ["connection pool", "exhausted", "connection limit"],
        }

        pattern_id = top_pattern.pattern_id.lower()
        keywords = pattern_keywords.get(pattern_id, [])

        if keywords and top_pattern.confidence > 0.7:
            pattern_mentioned = any(kw in answer_lower for kw in keywords)
            if not pattern_mentioned:
                issues.append(ValidationIssue(
                    issue_type="inconsistency",
                    severity="info",
                    description=(
                        f"Detected pattern '{top_pattern.pattern_name}' "
                        f"(confidence: {top_pattern.confidence:.0%}) "
                        f"is not clearly reflected in the LLM response."
                    ),
                ))

    # ── Check 3: Failure report alignment ──
    failures = intel_result.failure_report
    if failures.has_failures and failures.root_cause_candidates:
        # Check if any root cause candidate is mentioned
        root_funcs = []
        for fs in failures.failure_spans:
            if fs.is_root_cause:
                root_funcs.append(fs.span_function.lower())

        if root_funcs:
            any_root_mentioned = False
            for func in root_funcs:
                func_parts = re.split(r'[_\-./\s]+', func)
                meaningful = [p for p in func_parts if len(p) > 2]
                if any(part in answer_lower for part in meaningful):
                    any_root_mentioned = True
                    break

            if not any_root_mentioned:
                issues.append(ValidationIssue(
                    issue_type="inconsistency",
                    severity="info",
                    description=(
                        f"Root cause span(s) identified by failure analysis "
                        f"({', '.join(root_funcs[:3])}) are not mentioned in the response."
                    ),
                ))

    return issues


# ═══════════════════════════════════════════════════════════════════════════════
# DECISION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def _decide_action(issues: list[ValidationIssue]) -> tuple[str, float]:
    """
    Decide what to do based on the issues found.

    Returns:
        (action, confidence)
        action: "pass" | "caveat" | "fallback"
        confidence: 0.0-1.0
    """
    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")
    info_count = sum(1 for i in issues if i.severity == "info")

    # FALLBACK: Critical issues → replace with pattern explanation
    if error_count >= 1 or warning_count >= 4:
        confidence = max(0.1, 0.5 - (error_count * 0.2) - (warning_count * 0.1))
        return "fallback", confidence

    # CAVEAT: Minor issues → add notes to the answer
    if warning_count >= 1:
        confidence = max(0.3, 0.8 - (warning_count * 0.15) - (info_count * 0.05))
        return "caveat", confidence

    # PASS: Clean or only info-level notes
    if info_count > 0:
        confidence = max(0.7, 0.95 - (info_count * 0.05))
        return "pass", confidence

    # Perfect — no issues at all
    return "pass", 1.0


def _count_by_severity(issues: list[ValidationIssue]) -> str:
    """Format issue counts by severity for display."""
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")

    parts = []
    if errors:
        parts.append(f"{errors} error(s)")
    if warnings:
        parts.append(f"{warnings} warning(s)")
    if infos:
        parts.append(f"{infos} info note(s)")

    return ", ".join(parts) if parts else "none"


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def _add_caveats(answer: str, issues: list[ValidationIssue]) -> str:
    """Add caveat notes to the answer for minor issues."""
    warnings = [i for i in issues if i.severity == "warning"]

    if not warnings:
        return answer

    caveat_section = "\n\n---\n⚠️ **Validation Notes**:\n"

    for w in warnings:
        caveat_section += f"- {w.description}\n"

    return answer + caveat_section


def _build_fallback_answer(intel_result: IntelligenceResult) -> str:
    """Build a fallback answer from Intelligence Layer data when LLM answer is unreliable."""
    parts = []

    parts.append("## Analysis Results\n")
    parts.append(
        "*Note: The initial AI response could not be fully validated against the trace data. "
        "Below is an analysis based on the Intelligence Layer's findings.*\n"
    )

    # Pattern-based explanation (if available)
    if intel_result.pattern_matches:
        pattern = intel_result.pattern_matches[0]
        parts.append(f"### Detected Pattern: {pattern.pattern_name}")
        parts.append(f"**Confidence**: {pattern.confidence:.0%}\n")
        parts.append(f"**Explanation**: {pattern.explanation}\n")
        parts.append(f"**Recommended Fix**: {pattern.recommended_fix}\n")

    # Ranked causes
    ranked = intel_result.ranked_causes
    if ranked.causes:
        parts.append("### Ranked Causes")
        for cause in ranked.causes[:3]:
            parts.append(f"- **#{cause.rank} {cause.span_function}** (score: {cause.score:.2f})")
        parts.append("")

    # Failure report
    failures = intel_result.failure_report
    if failures.has_failures:
        parts.append("### Failures Detected")
        for fs in failures.failure_spans:
            root = " ← ROOT CAUSE" if fs.is_root_cause else ""
            parts.append(f"- **{fs.span_function}**: {fs.failure_type.value}{root}")
            if fs.error_messages:
                parts.append(f"  - Error: {fs.error_messages[0][:100]}")
        parts.append("")

    if not intel_result.pattern_matches and not ranked.causes:
        parts.append("No clear root cause could be determined from the available trace data.")

    return "\n".join(parts)
