import requests
from src.service.trace.log_client import LogClient
from src.models.log import LogEntry, TraceLogs
class JaegerLogClient(LogClient):
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        self.api_url = f"{jaeger_url}/api"
    async def get_logs_by_trace_id(self, trace_id: str) -> TraceLogs:
        response = requests.get(f"{self.api_url}/traces/{trace_id}")
        if response.status_code != 200:
            return TraceLogs(logs=[])
        
        data = response.json()
        if not data.get("data"):
            return TraceLogs(logs=[])
        
        trace_data = data["data"][0]
        spans_data = trace_data.get("spans", [])
        
        # Group logs by span_id
        logs_by_span = {}
        
        for span in spans_data:
            span_id = span["spanID"]
            span_logs = span.get("logs", [])
            
            for log_point in span_logs:
                # Convert Jaeger log to LogEntry
                timestamp = log_point["timestamp"] / 1_000_000  # micros to seconds
                
                # Extract fields
                fields = {f["key"]: f["value"] for f in log_point.get("fields", [])}
                
                # Determine level (fallback to INFO)
                level = fields.get("level", "INFO")
                if "level" in fields and isinstance(fields["level"], str):
                    level = fields["level"].upper()
                
                message = fields.get("message") or fields.get("event") or fields.get("msg") or ""
                
                entry = LogEntry(
                    time=timestamp,
                    level=level,
                    message=str(message),
                    file_name=fields.get("file", ""),
                    function_name=fields.get("function", ""),
                    line_number=int(fields.get("line", 0)),
                    span_id=span_id
                )
                
                logs_by_span.setdefault(span_id, []).append(entry)
        
        logs_list = []
        for span_id, entries in logs_by_span.items():
            entries.sort(key=lambda x: x.time)
            logs_list.append({span_id: entries})
            
        return TraceLogs(logs=logs_list)