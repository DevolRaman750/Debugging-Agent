import requests
from src.models.trace import Span, Trace
from src.service.trace.trace_client import TraceClient
class JaegerTraceClient(TraceClient):
    def __init__(self, jaeger_url: str = "http://localhost:16686"):
        self.api_url = f"{jaeger_url}/api"
    
    async def get_trace_by_id(self, trace_id: str) -> Trace | None:
        response = requests.get(f"{self.api_url}/traces/{trace_id}")
        if response.status_code != 200:
            return None
        
        data = response.json()
        if not data.get("data"):
            return None
        
        return self._convert_jaeger_trace(data["data"][0])
    
    async def get_services(self) -> list[str]:
        """Get all service names known to Jaeger."""
        response = requests.get(f"{self.api_url}/services")
        if response.status_code != 200:
            return []
        return response.json().get("data", [])

    async def get_recent_traces(self, start_time, end_time, limit: int = 50, service: str | None = None) -> list[Trace]:
        all_traces: list[Trace] = []

        if service:
            services = [service]
        else:
            # Query all known services
            services = await self.get_services()
            if not services:
                return []

        seen_trace_ids: set[str] = set()

        for svc in services:
            params = {
                "service": svc,
                "start": int(start_time.timestamp() * 1_000_000),
                "end": int(end_time.timestamp() * 1_000_000),
                "limit": limit,
            }

            response = requests.get(f"{self.api_url}/traces", params=params)
            if response.status_code != 200:
                continue

            data = response.json().get("data", [])

            for trace_data in data:
                tid = trace_data.get("traceID", "")
                if tid not in seen_trace_ids:
                    seen_trace_ids.add(tid)
                    trace = self._convert_jaeger_trace(trace_data)
                    # Set service name from the Jaeger service
                    if not trace.service_name:
                        trace.service_name = svc
                    all_traces.append(trace)

        # Sort by start_time descending (most recent first)
        all_traces.sort(key=lambda t: t.start_time, reverse=True)
        return all_traces[:limit]
    
    def _convert_jaeger_trace(self, jaeger_data: dict) -> Trace:
        spans_data = jaeger_data.get("spans", [])
        processes = jaeger_data.get("processes", {})
        spans_dict = {}

        # Determine primary service name from processes
        service_name = None
        if processes:
            # Use the first process's serviceName
            first_process = next(iter(processes.values()), {})
            service_name = first_process.get("serviceName")
        
        # Convert each span
        for span_data in spans_data:
            span = Span(
                id=span_data["spanID"],
                name=span_data["operationName"],
                start_time=span_data["startTime"] / 1_000_000,  # μs to s
                end_time=(span_data["startTime"] + span_data["duration"]) / 1_000_000,
                duration=span_data["duration"] / 1_000,  # μs to ms
            )
            spans_dict[span.id] = span
        
        # Build hierarchy
        root_spans = self._build_span_hierarchy(spans_data, spans_dict)
        
        return Trace(
            trace_id=jaeger_data["traceID"],
            start_time=min(s.start_time for s in spans_dict.values()),
            end_time=max(s.end_time for s in spans_dict.values()),
            duration=sum(s.duration for s in root_spans),
            service_name=service_name,
            spans=root_spans
        )

    
    
    def _build_span_hierarchy(self, spans_data, spans_dict) -> list[Span]:
        """Convert flat spans to nested tree using CHILD_OF references."""
        parent_map = {}
        
        for span_data in spans_data:
            span_id = span_data["spanID"]
            for ref in span_data.get("references", []):
                if ref.get("refType") == "CHILD_OF":
                    parent_map[span_id] = ref["spanID"]
                    break
        
        root_spans = []
        for span_id, span in spans_dict.items():
            if span_id in parent_map:
                parent_id = parent_map[span_id]
                if parent_id in spans_dict:
                    spans_dict[parent_id].spans.append(span)
            else:
                root_spans.append(span)
        
        return root_spans