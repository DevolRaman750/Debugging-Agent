from abc import ABC, abstractmethod
from src.models.trace import Trace
class TraceClient(ABC):
    @abstractmethod
    async def get_trace_by_id(self, trace_id: str) -> Trace | None:
        """Fetch a single trace with all spans."""
        pass
    
    @abstractmethod
    async def get_recent_traces(self, start_time, end_time, limit: int = 50) -> list[Trace]:
        """Fetch recent traces with pagination."""
        pass