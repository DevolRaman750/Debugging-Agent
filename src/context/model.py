from datetime import datetime
from pydantic import BaseModel,Field
class LogNode(BaseModel):
    log_utc_timestamp: datetime
    log_level: str
    log_file_name: str
    log_func_name: str
    log_message: str
    log_line_number: int
    
    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.log_utc_timestamp),
            "level": self.log_level,
            "message": self.log_message,
            "file": self.log_file_name,
            "line": self.log_line_number,
        }
class SpanNode(BaseModel):
    span_id: str
    func_full_name: str
    span_latency: float
    span_utc_start_time: datetime
    span_utc_end_time: datetime
    logs: list[LogNode] = Field(default_factory=list)
    children_spans: list["SpanNode"] = Field(default_factory=list)
    
    def to_dict(self) -> dict:
        # Interleave logs and children chronologically
        events = []
        for log in self.logs:
            events.append((log.log_utc_timestamp.timestamp(), "log", log))
        for child in self.children_spans:
            events.append((child.span_utc_start_time.timestamp(), "span", child))
        
        events.sort(key=lambda x: x[0])
        
        result = {
            "span_id": self.span_id,
            "function": self.func_full_name,
            "latency_ms": self.span_latency,
        }
        
        log_count = 0
        for _, event_type, obj in events:
            if event_type == "log":
                result[f"log_{log_count}"] = obj.to_dict()
                log_count += 1
            else:
                result[obj.span_id] = obj.to_dict()
        
        return result