from abc import ABC, abstractmethod
from src.models.log import TraceLogs
class LogClient(ABC):
    @abstractmethod
    async def get_logs_by_trace_id(self, trace_id: str) -> TraceLogs:
        """Fetch all logs for a trace, grouped by span_id."""
        pass