"""Debug: Check how Jaeger stores span events/logs."""
import requests
import json

# Get traces
from datetime import datetime, timedelta, timezone
end = datetime.now(timezone.utc)
start = end - timedelta(hours=3)
r = requests.get("http://localhost:8000/v1/explore/list-traces", params={
    "start_time": start.isoformat(),
    "end_time": end.isoformat(),
})
traces = r.json()["traces"]
# Pick an ecommerce trace
ecom = [t for t in traces if t["service_name"] == "ecommerce-backend-2"]
if ecom:
    tid = ecom[0]["trace_id"]
    print(f"Checking trace: {tid}")
    
    # Raw Jaeger data
    r2 = requests.get(f"http://localhost:16686/api/traces/{tid}")
    data = r2.json()["data"][0]
    
    for span in data["spans"]:
        op = span["operationName"]
        logs = span.get("logs", [])
        tags = span.get("tags", [])
        print(f"  Span '{op}': {len(logs)} logs, {len(tags)} tags")
        if logs:
            print(f"    Sample log: {json.dumps(logs[0], indent=4)}")
        if not logs and tags:
            # Check if events are stored as tags
            for tag in tags[:3]:
                print(f"    Tag: {tag['key']}={tag['value']}")
