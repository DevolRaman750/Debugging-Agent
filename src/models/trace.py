from pydantic import BaseModel,Field

class Span(BaseModel):
    id: str
    name: str
    start_time: float
    end_time: float
    duration: float
    spans: list['Span'] = []


class Trace(BaseModel):
    trace_id:str
    start_time: float
    end_time: float
    duration: float
    service_name: str |None = None
    spans: list[Span] = Field(default_factory=list)

    
