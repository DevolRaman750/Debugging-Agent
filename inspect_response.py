"""
inspect_response.py
====================
Runs 2 traces through SingleRCAAgent and pretty-prints the FULL
ChatbotResponse so you can verify every field the user receives.

  Trace A  → fast-path  (POST /api/checkout — Slow Query pattern)
  Trace B  → LLM path   (POST /api/payment — retry failures, no pattern)

Usage:
    .venv\Scripts\python inspect_response.py
"""

import asyncio, json, requests, time as timer
from datetime import datetime

from src.config import GROQ_API_KEY
from src.service.provider import ObservabilityProvider
from src.context.tree_builder import build_heterogeneous_tree
from src.context.utils import find_root_span
from src.routing.types import ChatModel
from src.agents.single_rca_agent import SingleRCAAgent


# ── Helpers ──────────────────────────────────────────────────────────────────

def separator(title: str):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def pretty(obj) -> str:
    """Pydantic model → indented JSON string."""
    return json.dumps(obj.model_dump(mode="json"), indent=2, default=str)


def print_field(label: str, value, indent: int = 0):
    pad = "  " * indent
    print(f"{pad}{label:.<40s} {value}")


# ── Core ─────────────────────────────────────────────────────────────────────

async def fetch_tree(provider, trace_id):
    trace = await provider.trace_client.get_trace_by_id(trace_id)
    logs = await provider.log_client.get_logs_by_trace_id(trace_id)
    root = find_root_span(trace.spans)
    return build_heterogeneous_tree(root, logs.logs)


async def run_one(agent, provider, trace_id, label):
    separator(f"{label}  (trace_id={trace_id[:16]}…)")

    tree = await fetch_tree(provider, trace_id)
    print(f"  Root span: {tree.func_full_name}  ({tree.span_latency:.1f} ms)")

    start = timer.time()
    response = await agent.chat(
        trace_id=trace_id,
        chat_id=f"inspect-{trace_id[:8]}",
        user_message="Why is this trace slow?",
        model=ChatModel.AUTO,
        timestamp=datetime.now(),
        tree=tree,
        chat_history=None,
        db_client=None,          # skip persistence for speed
    )
    elapsed = (timer.time() - start) * 1000

    # ── Print every field ──
    print(f"\n  ⏱  Round-trip: {elapsed:.0f} ms\n")

    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │               ChatbotResponse — Full Dump                  │")
    print("  └─────────────────────────────────────────────────────────────┘\n")

    print_field("chat_id", response.chat_id, 1)
    print_field("time", response.time.isoformat(), 1)
    print_field("message_type", response.message_type.value, 1)

    # Action
    if response.action:
        print_field("action.type", response.action.get("type"), 1)
        print_field("action.status", response.action.get("status"), 1)
    else:
        print_field("action", None, 1)

    # Validation
    print()
    print_field("validation_passed", response.validation_passed, 1)
    print_field("validation_confidence", response.validation_confidence, 1)
    print_field("fallback_used", response.fallback_used, 1)
    if response.validation_notes:
        for i, note in enumerate(response.validation_notes):
            print_field(f"  validation_notes[{i}]", note, 1)
    else:
        print_field("validation_notes", "[]", 1)

    # Metadata (IntelligenceMetadata)
    print()
    if response.metadata:
        m = response.metadata
        print_field("metadata.confidence", m.confidence, 1)
        print_field("metadata.pattern_matched", m.pattern_matched, 1)
        print_field("metadata.fast_path", m.fast_path, 1)
        print_field("metadata.processing_time_ms", f"{m.processing_time_ms:.1f}", 1)
        print_field("metadata.top_cause", m.top_cause, 1)
        print_field("metadata.top_cause_score", m.top_cause_score, 1)
        print_field("metadata.causes_found", m.causes_found, 1)
    else:
        print_field("metadata", None, 1)

    # References
    print()
    if response.reference:
        for ref in response.reference:
            print(f"    Reference #{ref.number}:")
            print_field("type", ref.type, 3)
            print_field("span_id", ref.span_id, 3)
            print_field("span_function_name", ref.span_function_name, 3)
            print_field("log_message", (ref.log_message or "")[:80], 3)
    else:
        print_field("references", "[] (none)", 1)

    # Message (truncated preview)
    print()
    preview = response.message[:300].replace("\n", "\n    ")
    print(f"    message (first 300 chars):\n    {preview}…" if len(response.message) > 300 else f"    message:\n    {response.message}")

    # Raw JSON dump
    print("\n  ── Raw JSON ─────────────────────────────────────────────────")
    print(pretty(response))

    return response


# ── Main ─────────────────────────────────────────────────────────────────────

async def main():
    separator("TraceRoot — Response Inspector")

    # Fetch all traces
    resp = requests.get(
        "http://localhost:16686/api/traces",
        params={"service": "ecommerce-backend-2", "limit": 20},
    )
    data = resp.json().get("data", [])
    if not data:
        print("❌ No traces found — run  python sample_app.py  first")
        return

    # Build lookup: root_operation → trace_id
    lookup = {}
    for t in data:
        for s in t["spans"]:
            if not s.get("references"):
                lookup[s["operationName"]] = t["traceID"]
                break

    # Pick one fast-path trace and one LLM trace
    fast_tid = lookup.get("POST /api/checkout")
    llm_tid = lookup.get("POST /api/payment")

    if not fast_tid or not llm_tid:
        print("⚠  Couldn't find expected traces. Available roots:")
        for op, tid in lookup.items():
            print(f"    {op:45s} → {tid[:16]}…")
        return

    provider = ObservabilityProvider.create_jaeger_provider()
    agent = SingleRCAAgent(groq_api_key=GROQ_API_KEY)

    # Run both
    print("\n  Will inspect 2 traces:")
    print(f"    A. ⚡ FAST PATH  — POST /api/checkout   ({fast_tid[:16]}…)")
    print(f"    B. 🤖 LLM PATH  — POST /api/payment    ({llm_tid[:16]}…)")

    await run_one(agent, provider, fast_tid, "A · ⚡ FAST PATH — POST /api/checkout")
    await run_one(agent, provider, llm_tid,  "B · 🤖 LLM  PATH — POST /api/payment")

    separator("Done — Inspect fields above to verify response shape")


if __name__ == "__main__":
    asyncio.run(main())
