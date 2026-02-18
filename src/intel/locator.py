import re
from src.intel.types import (
    ClassifiedSpanNode, FailureType, FailureSpan, FailureChain, FailureReport
)
ERROR_PATTERNS = {
    FailureType.TIMEOUT: [r"timeout", r"timed out", r"deadline exceeded"],
    FailureType.RATE_LIMIT: [r"rate limit", r"429", r"too many requests"],
    FailureType.AUTH_FAILURE: [r"401", r"403", r"unauthorized", r"forbidden"],
    FailureType.NOT_FOUND: [r"404", r"not found"],
    FailureType.INTERNAL_ERROR: [r"500", r"internal server error"],
    FailureType.EXCEPTION: [r"exception", r"error", r"traceback", r"failed"],
    FailureType.RESOURCE_EXHAUSTION: [r"oom", r"out of memory", r"connection pool"],
}
def detect_failure_type(logs: list) -> tuple[FailureType, list[str]]:
    """Detect failure type from logs."""
    error_messages = []
    
    for log in logs:
        msg = str(getattr(log, 'log_message', log)).lower()
        level = str(getattr(log, 'log_level', '')).upper()
        
        if level in ['ERROR', 'CRITICAL'] or 'error' in msg:
            error_messages.append(str(getattr(log, 'log_message', log)))
    
    if not error_messages:
        return FailureType.UNKNOWN, []
    
    # Match patterns
    all_text = " ".join(error_messages).lower()
    for failure_type, patterns in ERROR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, all_text):
                return failure_type, error_messages
    
    return FailureType.EXCEPTION, error_messages
def locate_failures(tree: ClassifiedSpanNode) -> FailureReport:
    """Find all failure spans and build failure chains."""
    failure_spans = []
    
    def traverse(span: ClassifiedSpanNode, depth: int = 0, ancestors: list[str] = None):
        ancestors = ancestors or []
        
        failure_type, error_msgs = detect_failure_type(span.logs)
        
        if error_msgs:
            # Check if any child also has errors (not root cause)
            child_has_error = any(
                detect_failure_type(c.logs)[1] for c in span.children_spans
            )
            
            failure_spans.append(FailureSpan(
                span_id=span.span_id,
                span_function=span.func_full_name,
                failure_type=failure_type,
                error_messages=error_msgs,
                is_root_cause=not child_has_error,
                depth_in_tree=depth,
            ))
        
        # Recurse
        for child in span.children_spans:
            traverse(child, depth + 1, ancestors + [span.span_id])
    
    traverse(tree)
    
    # Build failure chains
    failure_chains = []
    root_causes = [fs for fs in failure_spans if fs.is_root_cause]
    
    for rc in root_causes:
        chain_spans = [fs.span_id for fs in failure_spans if fs.depth_in_tree >= rc.depth_in_tree]
        if len(chain_spans) > 1:
            failure_chains.append(FailureChain(
                spans=chain_spans,
                propagation_path=" → ".join(chain_spans[:3])
            ))
    
    return FailureReport(
        failure_spans=failure_spans,
        failure_chains=failure_chains,
        root_cause_candidates=[fs.span_id for fs in root_causes],
        has_failures=len(failure_spans) > 0,
    )