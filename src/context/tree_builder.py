from datetime import datetime, timezone
from src.models.trace import Span
from src.models.log import LogEntry
from src.context.model import LogNode, SpanNode

def create_log_map(trace_logs:list[dict[str,list[LogEntry]]])->dict[str,list[LogEntry]]:
    logs_map = {}

    for logs_dict in trace_logs:
        for span_id, entries in logs_dict.items():
            logs_map.setdefault(span_id, []).extend(entries)
    return logs_map

def convert_log_entry_to_log_node(log: LogEntry) -> LogNode:
    return LogNode(
        log_utc_timestamp=datetime.fromtimestamp(log.time, tz=timezone.utc),
        log_level=log.level,
        log_file_name=log.file_name,
        log_func_name=log.function_name,
        log_message=log.message,
        log_line_number=log.line_number,
    )

def convert_span_to_span_node(span:Span,logs_map:dict[str, list[LogEntry]]) ->SpanNode:
    

    span_logs = [convert_log_entry_to_log_node(log) for log in logs_map.get(span.id, [])]
    span_logs.sort(key=lambda l: l.log_utc_timestamp)

    # Recursively convert children
    children = [convert_span_to_span_node(child, logs_map) for child in span.spans]
    children.sort(key=lambda c: c.span_utc_start_time)

    return SpanNode(
        span_id=span.id,
        func_full_name=span.name,
        span_latency=span.duration,
        span_utc_start_time=datetime.fromtimestamp(span.start_time, tz=timezone.utc),
        span_utc_end_time=datetime.fromtimestamp(span.end_time, tz=timezone.utc),
        logs=span_logs,
        children_spans=children,
    )


def build_heterogeneous_tree(span: Span, trace_logs: list[dict[str, list[LogEntry]]]) -> SpanNode:
    """Main entry point: Build heterogeneous tree from span + logs."""
    logs_map = create_log_map(trace_logs)
    return convert_span_to_span_node(span, logs_map)

