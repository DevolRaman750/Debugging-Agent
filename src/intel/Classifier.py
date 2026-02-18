import re
from src.intel.types import SpanType,ClassifiedSpanNode
from src.context.model import SpanNode



# Classification patterns
SPAN_TYPE_PATTERNS = {
    SpanType.LLM_CALL: [
        r"openai", r"anthropic", r"llm", r"chat.*completion",
        r"embedding", r"gpt", r"claude"
    ],
    SpanType.DB_QUERY: [
        r"sqlalchemy", r"psycopg", r"mysql", r"postgres", r"mongodb",
        r"execute", r"query", r"select", r"insert", r"update", r"delete"
    ],
    SpanType.HTTP_REQUEST: [
        r"requests\.", r"httpx", r"aiohttp", r"urllib", r"fetch"
    ],
    SpanType.HTTP_HANDLER: [
        r"fastapi", r"flask", r"django", r"starlette", r"endpoint"
    ],
    SpanType.CACHE_OP: [
        r"redis", r"memcached", r"cache\.get", r"cache\.set"
    ],
    SpanType.QUEUE_PUBLISH: [
        r"kafka.*produce", r"rabbitmq.*publish", r"sqs.*send"
    ],
    SpanType.QUEUE_CONSUME: [
        r"kafka.*consume", r"rabbitmq.*consume", r"sqs.*receive"
    ],
    SpanType.HEALTH_CHECK: [
        r"health", r"liveness", r"readiness", r"ping", r"heartbeat"
    ],
    SpanType.RETRY_LOOP: [
        r"retry", r"attempt", r"backoff"
    ],
}
def classify_span_type(func_name: str, logs: list) -> tuple[SpanType, float]:
    """Classify a span based on function name and logs."""
    func_lower = func_name.lower()
    
    # Check function name patterns
    for span_type, patterns in SPAN_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, func_lower):
                return span_type, 0.9
    
    # Check log messages
    log_text = " ".join(str(log) for log in logs).lower()
    for span_type, patterns in SPAN_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, log_text):
                return span_type, 0.7
    
    return SpanType.BUSINESS_LOGIC, 0.5
def classify_spans(tree: SpanNode) -> ClassifiedSpanNode:
    """Recursively classify all spans in the tree."""
    span_type, confidence = classify_span_type(tree.func_full_name, tree.logs)
    
    classified_children = [
        classify_spans(child) for child in tree.children_spans
    ]
    
    return ClassifiedSpanNode(
        span_id=tree.span_id,
        func_full_name=tree.func_full_name,
        span_latency=tree.span_latency,
        span_utc_start_time=tree.span_utc_start_time,
        span_utc_end_time=tree.span_utc_end_time,
        logs=tree.logs,
        children_spans=classified_children,
        span_type=span_type,
        classification_confidence=confidence,
    )