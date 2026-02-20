from src.intel.types import (
    ClassifiedSpanNode, FailureReport, PatternMatch,
    RankedCause, RankedCauses, ScoringFactor, Evidence,
    ConfidenceLevel, RankingConfig
)
def calculate_error_density(span: ClassifiedSpanNode) -> float:
    """Calculate ratio of error logs to total logs."""
    if not span.logs:
        return 0.0
    error_count = sum(1 for log in span.logs 
                     if hasattr(log, 'log_level') and log.log_level in ['ERROR', 'CRITICAL'])
    return error_count / len(span.logs)
def calculate_latency_anomaly(span: ClassifiedSpanNode, baseline_ms: float = 100) -> float:
    """Score based on how much latency exceeds baseline."""
    if span.span_latency <= baseline_ms:
        return 0.0
    return min(1.0, (span.span_latency - baseline_ms) / (baseline_ms * 10))
def rank_causes(
    tree: ClassifiedSpanNode,
    failure_report: FailureReport,
    pattern_matches: list[PatternMatch],
    config: RankingConfig = None
) -> RankedCauses:
    """Score and rank candidate causes."""
    config = config or RankingConfig()
    
    candidates = []
    
    # Get all failure spans as candidates
    span_lookup = {}
    def build_lookup(span: ClassifiedSpanNode, depth: int = 0):
        span_lookup[span.span_id] = (span, depth)
        for child in span.children_spans:
            build_lookup(child, depth + 1)
    build_lookup(tree)
    
    for fs in failure_report.failure_spans:
        if fs.span_id not in span_lookup:
            continue
        
        span, depth = span_lookup[fs.span_id]
        
        # Calculate scoring factors
        error_density = calculate_error_density(span)
        latency_anomaly = calculate_latency_anomaly(span)
        depth_score = min(1.0, depth / 5.0)  # Deeper = more likely root cause
        
        # Check if matched by pattern
        pattern_score = 0.0
        for pm in pattern_matches:
            if fs.span_id in pm.matched_spans:
                pattern_score = max(pattern_score, pm.confidence)
        
        # Calculate weighted score
        score = (
            error_density * config.error_density_weight +
            latency_anomaly * config.latency_anomaly_weight +
            depth_score * config.depth_weight +
            pattern_score * config.pattern_match_weight
        )
        
        factors = [
            ScoringFactor(factor_name="error_density", factor_value=error_density,
                         weighted_contribution=error_density * config.error_density_weight),
            ScoringFactor(factor_name="latency_anomaly", factor_value=latency_anomaly,
                         weighted_contribution=latency_anomaly * config.latency_anomaly_weight),
            ScoringFactor(factor_name="depth", factor_value=depth_score,
                         weighted_contribution=depth_score * config.depth_weight),
            ScoringFactor(factor_name="pattern_match", factor_value=pattern_score,
                         weighted_contribution=pattern_score * config.pattern_match_weight),
        ]
        
        evidence = [Evidence(
            span_id=fs.span_id,
            evidence_type="error_log",
            description=msg[:100],
        ) for msg in fs.error_messages[:3]]
        
        candidates.append(RankedCause(
            span_id=fs.span_id,
            span_function=fs.span_function,
            score=score,
            rank=0,  # Will be set after sorting
            contributing_factors=factors,
            evidence=evidence,
        ))
    
    # Also include pattern-matched spans that aren't already failure spans.
    # This ensures patterns like N+1 queries (no errors, just anti-patterns)
    # still produce ranked candidates for the fast path response.
    already_added = {c.span_id for c in candidates}
    
    for pm in pattern_matches:
        for matched_span_id in pm.matched_spans:
            if matched_span_id in already_added or matched_span_id not in span_lookup:
                continue
            
            span, depth = span_lookup[matched_span_id]
            
            error_density = calculate_error_density(span)
            latency_anomaly = calculate_latency_anomaly(span)
            depth_score = min(1.0, depth / 5.0)
            pattern_score = pm.confidence
            
            score = (
                error_density * config.error_density_weight +
                latency_anomaly * config.latency_anomaly_weight +
                depth_score * config.depth_weight +
                pattern_score * config.pattern_match_weight
            )
            
            factors = [
                ScoringFactor(factor_name="error_density", factor_value=error_density,
                             weighted_contribution=error_density * config.error_density_weight),
                ScoringFactor(factor_name="latency_anomaly", factor_value=latency_anomaly,
                             weighted_contribution=latency_anomaly * config.latency_anomaly_weight),
                ScoringFactor(factor_name="depth", factor_value=depth_score,
                             weighted_contribution=depth_score * config.depth_weight),
                ScoringFactor(factor_name="pattern_match", factor_value=pattern_score,
                             weighted_contribution=pattern_score * config.pattern_match_weight),
            ]
            
            candidates.append(RankedCause(
                span_id=matched_span_id,
                span_function=span.func_full_name,
                score=score,
                rank=0,
                contributing_factors=factors,
                evidence=[Evidence(
                    span_id=matched_span_id,
                    evidence_type="pattern_match",
                    description=f"Matched pattern: {pm.pattern_name}",
                )],
            ))
            already_added.add(matched_span_id)
    
    # Sort by score and assign ranks
    candidates.sort(key=lambda c: c.score, reverse=True)
    for i, c in enumerate(candidates):
        c.rank = i + 1
    
    # Determine confidence level
    if not candidates:
        confidence = ConfidenceLevel.LOW
        top_score = 0.0
        score_gap = 0.0
    else:
        top_score = candidates[0].score
        score_gap = top_score - candidates[1].score if len(candidates) > 1 else top_score
        
        if top_score > 0.8 and score_gap > 0.3:
            confidence = ConfidenceLevel.HIGH
        elif top_score > 0.6:
            confidence = ConfidenceLevel.MEDIUM
        else:
            confidence = ConfidenceLevel.LOW
    
    return RankedCauses(
        causes=candidates,
        confidence_level=confidence,
        top_cause_score=top_score,
        score_gap=score_gap,
    )