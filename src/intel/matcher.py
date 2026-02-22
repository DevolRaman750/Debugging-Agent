from src.intel.types import ClassifiedSpanNode, FailureReport, PatternMatch
from src.intel.patterns.database import N1QueryPattern, SlowQueryPattern

# Register all patterns
PATTERNS = [
    N1QueryPattern(),
    SlowQueryPattern(),
    # Add more patterns here
]

def match_patterns(
    tree: ClassifiedSpanNode,
    failure_report: FailureReport,
    user_query: str = "",
    pattern_confidence_overrides: dict[str, float] | None = None,
) -> list[PatternMatch]:
    """Run all patterns against the tree.

    Args:
        tree:                         Classified span tree
        failure_report:               Failures from locator
        user_query:                   User's question
        pattern_confidence_overrides: Pattern name → multiplier (0.0-1.0)
            from eval_config.json.  Applied after the raw match confidence.
            Missing entries → 1.0 (no change).
    """
    overrides = pattern_confidence_overrides or {}
    matches = []

    for pattern in PATTERNS:
        match = pattern.match(tree, failure_report)
        if match:
            # Apply feedback-adjusted confidence multiplier
            multiplier = overrides.get(match.pattern_name, 1.0)
            if multiplier < 1.0:
                match.confidence = round(match.confidence * multiplier, 4)
            # Disable pattern entirely if multiplier is 0
            if multiplier > 0:
                matches.append(match)

    # Sort by confidence descending
    matches.sort(key=lambda m: m.confidence, reverse=True)

    return matches