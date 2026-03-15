"""
TraceRoot REST API Server
═════════════════════════
Serves trace and log data from Jaeger to the Next.js frontend.

Endpoints:
  GET /v1/explore/list-traces          — list recent traces in a time range
  GET /v1/explore/get-logs-by-trace-id — get span-level logs for a trace

Run:
  python -m src.api.server
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.service.provider import ObservabilityProvider

app = FastAPI(title="TraceRoot API", version="0.1.0")

# Allow the Next.js dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Shared provider instance (Jaeger)
_provider: ObservabilityProvider | None = None


def get_provider() -> ObservabilityProvider:
    global _provider
    if _provider is None:
        _provider = ObservabilityProvider.create_jaeger_provider()
    return _provider


# ── GET /v1/explore/list-traces ───────────────────────────────

@app.get("/v1/explore/list-traces")
async def list_traces(
    start_time: str = Query(..., description="ISO-8601 start"),
    end_time: str = Query(..., description="ISO-8601 end"),
    trace_provider: str = Query("jaeger"),
    log_provider: str = Query("jaeger"),
    trace_region: Optional[str] = Query(None),
    log_region: Optional[str] = Query(None),
    trace_id: Optional[str] = Query(None),
    pagination_token: Optional[str] = Query(None),
    categories: Optional[list[str]] = Query(None),
    values: Optional[list[str]] = Query(None),
    operations: Optional[list[str]] = Query(None),
    limit: int = Query(50, ge=1, le=200),
):
    """Return traces from Jaeger within the given time window."""
    provider = get_provider()

    # If a specific trace_id is requested, fetch just that trace
    if trace_id:
        trace = await provider.trace_client.get_trace_by_id(trace_id)
        if trace:
            trace_dict = trace.model_dump()
            # Enrich with service name from Jaeger
            _enrich_trace(trace_dict, trace_id)
            return {
                "traces": [trace_dict],
                "has_more": False,
                "next_pagination_token": None,
            }
        return {"traces": [], "has_more": False, "next_pagination_token": None}

    # Parse time range
    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid time format. Use ISO-8601."},
        )

    # Fetch recent traces from Jaeger
    traces = await provider.trace_client.get_recent_traces(
        start_time=start_dt,
        end_time=end_dt,
        limit=limit,
    )

    # Apply search filters if provided
    if categories and values and operations:
        traces = _apply_search_filters(traces, categories, values, operations)

    # Convert to dicts
    trace_dicts = []
    for t in traces:
        d = t.model_dump()
        _enrich_trace(d, d.get("trace_id", ""))
        trace_dicts.append(d)

    return {
        "traces": trace_dicts,
        "has_more": False,
        "next_pagination_token": None,
    }


def _enrich_trace(trace_dict: dict, trace_id: str) -> None:
    """Add service_name from spans if not already set."""
    if not trace_dict.get("service_name"):
        # Use the root span name as the service name
        spans = trace_dict.get("spans", [])
        if spans:
            trace_dict["service_name"] = spans[0].get("name", "unknown")
        else:
            trace_dict["service_name"] = "unknown"


def _apply_search_filters(traces, categories, values, operations):
    """Filter traces based on search criteria triplets."""
    filtered = []
    for trace in traces:
        match = True
        for cat, val, op in zip(categories, values, operations):
            if not _matches_criterion(trace, cat, val, op):
                match = False
                break
        if match:
            filtered.append(trace)
    return filtered


def _matches_criterion(trace, category: str, value: str, operation: str) -> bool:
    """Check if a trace matches a single search criterion."""
    target = ""
    if category == "service":
        target = trace.service_name or ""
    elif category == "operation":
        # Check span names
        target = " ".join(_collect_span_names(trace.spans))
    elif category == "trace_id":
        target = trace.trace_id
    elif category == "log":
        # Search through all span names and trace_id
        target = f"{trace.trace_id} {trace.service_name or ''} {' '.join(_collect_span_names(trace.spans))}"
    else:
        target = f"{trace.trace_id} {trace.service_name or ''}"

    value_lower = value.lower()
    target_lower = target.lower()

    if operation == "contains":
        return value_lower in target_lower
    elif operation == "not_contains":
        return value_lower not in target_lower
    elif operation == "equals":
        return value_lower == target_lower
    elif operation == "not_equals":
        return value_lower != target_lower
    elif operation == "starts_with":
        return target_lower.startswith(value_lower)
    elif operation == "ends_with":
        return target_lower.endswith(value_lower)
    return True


def _collect_span_names(spans) -> list[str]:
    """Recursively collect all span names from a span tree."""
    names = []
    for span in spans:
        names.append(span.name)
        if span.spans:
            names.extend(_collect_span_names(span.spans))
    return names


# ── GET /v1/explore/get-logs-by-trace-id ──────────────────────

@app.get("/v1/explore/get-logs-by-trace-id")
async def get_logs_by_trace_id(
    trace_id: str = Query(..., description="The trace ID"),
    trace_provider: str = Query("jaeger"),
    log_provider: str = Query("jaeger"),
    start_time: Optional[str] = Query(None),
    end_time: Optional[str] = Query(None),
    log_group_name: Optional[str] = Query(None),
    trace_region: Optional[str] = Query(None),
    log_region: Optional[str] = Query(None),
):
    """Return span-level logs for a given trace."""
    provider = get_provider()
    trace_logs = await provider.log_client.get_logs_by_trace_id(trace_id)

    return {
        "logs": trace_logs.model_dump(),
    }


# ── Health ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("  TraceRoot REST API Server")
    print("  http://localhost:8000")
    print("  Jaeger backend: http://localhost:16686")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000)
