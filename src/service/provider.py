from src.service.trace.jaeger_client import JaegerTraceClient
from src.service.log.jaeger_log_client import JaegerLogClient
class ObservabilityProvider:
    def __init__(self, trace_client, log_client):
        self.trace_client = trace_client
        self.log_client = log_client
    
    @classmethod
    def create_jaeger_provider(cls, jaeger_url: str = "http://localhost:16686"):
        return cls(
            trace_client=JaegerTraceClient(jaeger_url),
            log_client=JaegerLogClient(jaeger_url)
        )