from src.intel.types import (
    ClassifiedSpanNode, SpanType, SuppressionStats, SuppressionConfig
)
def has_errors(span: ClassifiedSpanNode) -> bool:
    """Check if span has ERROR or CRITICAL logs."""
    for log in span.logs:
        if hasattr(log, 'log_level') and log.log_level in ['ERROR', 'CRITICAL']:
            return True
    return False
def count_spans(span: ClassifiedSpanNode) -> int:
    """Recursively count all spans in tree."""
    count = 1
    for child in span.children_spans:
        count += count_spans(child)
    return count
def suppress_noise(
    tree: ClassifiedSpanNode,
    config: SuppressionConfig = None
) -> tuple[ClassifiedSpanNode, SuppressionStats]:
    """Remove uninformative spans to reduce context size."""
    config = config or SuppressionConfig()
    original_count = count_spans(tree)
    
    suppressed = 0
    collapsed = 0
    
    def should_suppress(span: ClassifiedSpanNode) -> bool:
        nonlocal suppressed
        
        # Never suppress spans with errors
        if has_errors(span):
            return False
        
        # Never suppress LLM calls
        if span.span_type == SpanType.LLM_CALL:
            return False
        
        # Suppress health checks
        if config.suppress_health_checks and span.span_type == SpanType.HEALTH_CHECK:
            suppressed += 1
            return True
        
        # Suppress instrumentation
        if config.suppress_instrumentation and span.span_type == SpanType.INSTRUMENTATION:
            suppressed += 1
            return True
        
        # Suppress very fast spans with no children
        if span.span_latency < config.min_duration_ms and len(span.children_spans) == 0:
            suppressed += 1
            return True
        
        return False
    
    def filter_children(span: ClassifiedSpanNode) -> ClassifiedSpanNode:
        """Recursively filter children."""
        filtered = []
        for child in span.children_spans:
            if not should_suppress(child):
                filtered.append(filter_children(child))
        
        return ClassifiedSpanNode(
            span_id=span.span_id,
            func_full_name=span.func_full_name,
            span_latency=span.span_latency,
            span_utc_start_time=span.span_utc_start_time,
            span_utc_end_time=span.span_utc_end_time,
            logs=span.logs,
            children_spans=filtered,
            span_type=span.span_type,
            classification_confidence=span.classification_confidence,
        )
    
    pruned_tree = filter_children(tree)
    remaining_count = count_spans(pruned_tree)
    
    stats = SuppressionStats(
        original_span_count=original_count,
        remaining_span_count=remaining_count,
        suppressed_count=suppressed,
        collapsed_count=collapsed,
        compression_ratio=original_count / max(remaining_count, 1)
    )
    
    return pruned_tree, stats