import json
from src.intel.types import (
    ClassifiedSpanNode, RankedCauses, CompressedContext, Evidence
)
def estimate_tokens(text: str) -> int:
    """Estimate token count (rough: ~4 chars per token)."""
    return len(text) // 4
def compress_context(
    tree: ClassifiedSpanNode,
    ranked_causes: RankedCauses,
    token_budget: int = 5000
) -> CompressedContext:
    """Compress tree to fit within token budget."""
    
    # Get top cause span_ids
    top_causes = set(c.span_id for c in ranked_causes.causes[:3])
    
    def compress_span(span: ClassifiedSpanNode, budget: int, is_top_cause: bool) -> dict:
        """Recursively compress a span."""
        
        # Always include basic info
        result = {
            "span_id": span.span_id,
            "function": span.func_full_name,
            "latency_ms": span.span_latency,
            "type": span.span_type.value,
        }
        
        # Include full logs for top causes, summarize for others
        if is_top_cause or span.span_id in top_causes:
            result["logs"] = [
                {"level": getattr(l, 'log_level', 'INFO'), 
                 "message": str(getattr(l, 'log_message', l))[:200]}
                for l in span.logs[:10]
            ]
        elif span.logs:
            error_count = sum(1 for l in span.logs 
                            if getattr(l, 'log_level', '') in ['ERROR', 'CRITICAL'])
            result["log_summary"] = f"{len(span.logs)} logs ({error_count} errors)"
        
        # Compress children
        if span.children_spans:
            child_budget = budget // max(len(span.children_spans), 1)
            result["children"] = [
                compress_span(c, child_budget, span.span_id in top_causes)
                for c in span.children_spans
            ]
        
        return result
    
    compressed_tree = compress_span(tree, token_budget, tree.span_id in top_causes)
    
    # Calculate token counts
    original_json = json.dumps(tree.model_dump(), default=str)
    compressed_json = json.dumps(compressed_tree, default=str)
    
    original_tokens = estimate_tokens(original_json)
    compressed_tokens = estimate_tokens(compressed_json)
    
    # Preserve evidence from ranked causes
    preserved = []
    for cause in ranked_causes.causes[:5]:
        preserved.extend(cause.evidence)
    
    return CompressedContext(
        compressed_tree=compressed_tree,
        token_count=compressed_tokens,
        original_token_count=original_tokens,
        compression_ratio=original_tokens / max(compressed_tokens, 1),
        preserved_evidence=preserved,
        truncation_notes=[
            f"Compressed from {original_tokens} to {compressed_tokens} tokens",
            f"Preserved top {len(top_causes)} cause spans in full detail",
        ],
    )