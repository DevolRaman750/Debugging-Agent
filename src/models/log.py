from pydantic import BaseModel


class LogEntry(BaseModel):
    time: float                  
    level: str                   
    message: str
    file_name: str = ""
    function_name: str = ""
    line_number: int = 0
    span_id: str = ""

    
class TraceLogs(BaseModel):
    logs: list[dict[str, list[LogEntry]]] = []  