# src/context/utils.py

from src.models.trace import Span

def find_root_span(spans: list[Span]) -> Span:
    """
    Find the true root span (the one that is never a child of any other span).
    """
    all_children = set()

    for span in spans:
        for child in span.spans:
            all_children.add(child.id)

    for span in spans:
        if span.id not in all_children:
            return span

    # Fallback safety
    return spans[0]
