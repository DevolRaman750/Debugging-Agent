"""Quick test: verify that the FastAPI server returns traces from Jaeger."""
import requests
from datetime import datetime, timedelta, timezone

end = datetime.now(timezone.utc)
start = end - timedelta(hours=3)

r = requests.get(
    "http://localhost:8000/v1/explore/list-traces",
    params={
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "trace_provider": "jaeger",
        "log_provider": "jaeger",
    },
)

data = r.json()
print(f"Status: {r.status_code}")
traces = data.get("traces", [])
print(f"Traces: {len(traces)}")

for t in traces:
    tid = t["trace_id"][:16]
    svc = t.get("service_name", "?")
    spans = len(t.get("spans", []))
    dur = t["duration"]
    print(f"  - {tid}... service={svc} root_spans={spans} dur={dur:.1f}ms")

# Test logs endpoint with first trace
if traces:
    tid = traces[0]["trace_id"]
    r2 = requests.get(
        "http://localhost:8000/v1/explore/get-logs-by-trace-id",
        params={"trace_id": tid, "trace_provider": "jaeger", "log_provider": "jaeger"},
    )
    log_data = r2.json()
    logs = log_data.get("logs", {}).get("logs", [])
    print(f"\nLogs for trace {tid[:16]}...: {len(logs)} span groups")
    for span_group in logs[:3]:
        for span_id, entries in span_group.items():
            print(f"  span {span_id[:12]}...: {len(entries)} log entries")
