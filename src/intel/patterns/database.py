from src.intel.patterns.base import FailurePattern
from src.intel.types import ClassifiedSpanNode, FailureReport, PatternMatch, SpanType
class N1QueryPattern(FailurePattern):
    """Detects N+1 query anti-pattern."""
    
    pattern_id = "n_plus_1_query"
    pattern_name = "N+1 Query"
    pattern_category = "database"
    
    def match(self, tree: ClassifiedSpanNode, failure_report: FailureReport) -> PatternMatch | None:
        # Find repeated DB queries under same parent
        db_spans = []
        
        def find_db_spans(span: ClassifiedSpanNode, parent_id: str = None):
            if span.span_type == SpanType.DB_QUERY:
                db_spans.append((span, parent_id))
            for child in span.children_spans:
                find_db_spans(child, span.span_id)
        
        find_db_spans(tree)
        
        # Group by parent
        by_parent = {}
        for span, parent in db_spans:
            by_parent.setdefault(parent, []).append(span)
        
        # Check for N+1 pattern (3+ similar queries under same parent)
        for parent_id, spans in by_parent.items():
            if len(spans) >= 3:
                return PatternMatch(
                    pattern_id=self.pattern_id,
                    pattern_name=self.pattern_name,
                    pattern_category=self.pattern_category,
                    confidence=min(0.9, 0.5 + len(spans) * 0.1),
                    matched_spans=[s.span_id for s in spans],
                    matched_evidence=[s.func_full_name for s in spans[:3]],
                    explanation=f"Found {len(spans)} individual database queries that could be batched into a single query.",
                    recommended_fix="Use batch queries or JOINs instead of individual queries in a loop.",
                )
        
        return None
class SlowQueryPattern(FailurePattern):
    """Detects slow database queries."""
    
    pattern_id = "slow_query"
    pattern_name = "Slow Query"
    pattern_category = "database"
    
    SLOW_THRESHOLD_MS = 1000  # 1 second
    
    def match(self, tree: ClassifiedSpanNode, failure_report: FailureReport) -> PatternMatch | None:
        slow_spans = []
        
        def find_slow(span: ClassifiedSpanNode):
            if span.span_type == SpanType.DB_QUERY and span.span_latency > self.SLOW_THRESHOLD_MS:
                slow_spans.append(span)
            for child in span.children_spans:
                find_slow(child)
        
        find_slow(tree)
        
        if slow_spans:
            slowest = max(slow_spans, key=lambda s: s.span_latency)
            return PatternMatch(
                pattern_id=self.pattern_id,
                pattern_name=self.pattern_name,
                pattern_category=self.pattern_category,
                confidence=min(0.95, 0.6 + slowest.span_latency / 5000),
                matched_spans=[s.span_id for s in slow_spans],
                matched_evidence=[f"{s.func_full_name}: {s.span_latency}ms" for s in slow_spans],
                explanation=f"Database query took {slowest.span_latency}ms. Consider adding indexes or optimizing the query.",
                recommended_fix="Add database indexes, optimize query, or add caching.",
            )
        
        return None