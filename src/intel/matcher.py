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
    user_query: str = ""
) -> list[PatternMatch]:
    """Run all patterns against the tree."""
    matches = []
    
    for pattern in PATTERNS:
        match = pattern.match(tree, failure_report)
        if match:
            matches.append(match)
    
    # Sort by confidence descending
    matches.sort(key=lambda m: m.confidence, reverse=True)
    
    return matches