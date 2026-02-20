# test_all_traces.py
# Fetches ALL traces from Jaeger for ecommerce-backend-2 and runs the
# intelligence pipeline, ChatRouter, and SingleRCAAgent on each one.

import asyncio
import requests
import time as timer
from datetime import datetime

from src.config import GROQ_API_KEY
from src.service.provider import ObservabilityProvider
from src.context.tree_builder import build_heterogeneous_tree
from src.context.utils import find_root_span
from src.intel.pipeline import IntelligencePipeline
from src.routing.types import ChatMode, ChatModel
from src.routing.router import ChatRouter
from src.agents.single_rca_agent import SingleRCAAgent


def print_section(title: str, width: int = 70):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_classified_tree(node, indent=0):
    prefix = "  " * indent
    print(f"{prefix}[{node.span_type.value}] {node.func_full_name} "
          f"({node.span_latency:.1f}ms, conf={node.classification_confidence:.2f}, "
          f"logs={len(node.logs)})")
    for child in node.children_spans:
        print_classified_tree(child, indent + 1)


async def test_trace(provider, trace_id, scenario_label):
    """Run full intelligence pipeline on a single trace."""
    print_section(f"{scenario_label}  |  trace_id={trace_id[:16]}...")

    # Fetch trace
    trace = await provider.trace_client.get_trace_by_id(trace_id)
    if not trace:
        print(f"  ❌ Trace not found!")
        return

    # Fetch logs
    logs = await provider.log_client.get_logs_by_trace_id(trace_id)

    # Build tree
    root_span = find_root_span(trace.spans)
    tree = build_heterogeneous_tree(root_span, logs.logs)

    print(f"  Root: {tree.func_full_name}  ({tree.span_latency:.1f}ms)")
    print(f"  Spans: {len(trace.spans)}")

    # Run intelligence pipeline
    pipeline = IntelligencePipeline()
    result = await pipeline.process(tree=tree, user_query="Why is this trace slow?")

    # ── Classification ──
    print(f"\n  📊 Classification Tree:")
    print_classified_tree(result.classified_tree, indent=2)

    # ── Suppression ──
    stats = result.suppression_stats
    print(f"\n  🔇 Suppression: {stats.suppressed_count}/{stats.original_span_count} suppressed "
          f"({stats.compression_ratio:.1f}x)")

    # ── Failures ──
    fr = result.failure_report
    if fr.has_failures:
        print(f"\n  🔴 Failures: {len(fr.failure_spans)} found")
        for fs in fr.failure_spans:
            rc = "ROOT CAUSE" if fs.is_root_cause else "symptom"
            print(f"      [{fs.failure_type.value}] {fs.span_function} ({rc})")
            if fs.error_messages:
                print(f"        → {fs.error_messages[0][:80]}")
    else:
        print(f"\n  🟢 No failures detected")

    # ── Patterns ──
    if result.pattern_matches:
        print(f"\n  🔍 Pattern Matches:")
        for pm in result.pattern_matches:
            print(f"      {pm.pattern_name} (confidence={pm.confidence:.2f})")
            print(f"        {pm.explanation[:80]}")
            print(f"        Fix: {pm.recommended_fix[:80]}")
            print(f"        Matched spans: {len(pm.matched_spans)}")
    else:
        print(f"\n  ⚪ No patterns matched")

    # ── Ranked Causes ──
    ranked = result.ranked_causes
    print(f"\n  📈 Ranked Causes: confidence={ranked.confidence_level.value}, "
          f"top_score={ranked.top_cause_score:.2f}, gap={ranked.score_gap:.2f}")
    for c in ranked.causes[:3]:
        print(f"      #{c.rank} {c.span_function} (score={c.score:.3f})")

    # ── Fast Path ──
    if result.fast_path_available:
        print(f"\n  ⚡ FAST PATH AVAILABLE — skip LLM!")
        print(f"     Pattern: {result.pattern_matches[0].pattern_name}")
        print(f"     Explanation: {result.pattern_matches[0].explanation}")
    else:
        print(f"\n  🐢 Fast path NOT available — needs LLM reasoning")

    return result, tree, trace_id


async def test_routing(router, query, has_trace):
    """Test ChatRouter routing decision for a query."""
    route = await router.route_query(
        user_message=query,
        chat_mode=ChatMode.AGENT,
        has_trace_context=has_trace,
    )
    return route


async def test_rca_agent(rca_agent, trace_id, tree, user_message):
    """Test SingleRCAAgent end-to-end (fast path or LLM)."""
    start = timer.time()
    response = await rca_agent.chat(
        trace_id=trace_id,
        chat_id=f"test-{trace_id[:8]}",
        user_message=user_message,
        model=ChatModel.AUTO,
        timestamp=datetime.now(),
        tree=tree,
        chat_history=None,
    )
    elapsed = (timer.time() - start) * 1000
    return response, elapsed


async def main():
    print("=" * 70)
    print("  TraceRoot — Test ALL Traces from Jaeger")
    print("  Service: ecommerce-backend-2")
    print("=" * 70)

    # Step 1: Fetch all trace IDs from Jaeger
    print("\nFetching traces from Jaeger...")
    resp = requests.get(
        "http://localhost:16686/api/traces",
        params={"service": "ecommerce-backend-2", "limit": 20}
    )
    data = resp.json().get("data", [])

    if not data:
        print("❌ No traces found for service 'ecommerce-backend-2'")
        print("   Run 'python sample_app.py' first!")
        return

    # Build list of (trace_id, root_operation, span_count)
    traces_info = []
    for t in data:
        tid = t["traceID"]
        root_op = "?"
        for s in t["spans"]:
            if not s.get("references"):
                root_op = s["operationName"]
                break
        traces_info.append((tid, root_op, len(t["spans"])))

    print(f"\n✅ Found {len(traces_info)} traces:\n")
    for i, (tid, op, cnt) in enumerate(traces_info, 1):
        print(f"  {i}. {tid[:16]}...  {op:45s} ({cnt} spans)")

    # Step 2: Test each trace through Intelligence Pipeline
    provider = ObservabilityProvider.create_jaeger_provider()

    results_summary = []
    trace_trees = []  # Store (trace_id, tree, intel_result) for RCA tests
    for i, (tid, op, cnt) in enumerate(traces_info, 1):
        label = f"Trace {i}/{len(traces_info)}: {op}"
        result, tree, trace_id = await test_trace(provider, tid, label)
        if result:
            results_summary.append({
                "trace_id": tid[:16],
                "root_op": op,
                "fast_path": result.fast_path_available,
                "patterns": len(result.pattern_matches),
                "failures": len(result.failure_report.failure_spans),
                "top_score": result.ranked_causes.top_cause_score,
            })
            trace_trees.append((tid, op, tree, result))

    # Step 3: Intelligence Pipeline Summary
    print_section("INTELLIGENCE PIPELINE SUMMARY")
    print(f"  {'#':<3} {'Root Operation':<40} {'Fail':>4} {'Pat':>4} {'Fast':>5} {'Score':>6}")
    print(f"  {'─'*3} {'─'*40} {'─'*4} {'─'*4} {'─'*5} {'─'*6}")
    for i, r in enumerate(results_summary, 1):
        fast = "⚡YES" if r["fast_path"] else "  no"
        print(f"  {i:<3} {r['root_op']:<40} {r['failures']:>4} {r['patterns']:>4} {fast:>5} {r['top_score']:>6.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 4: Test ChatRouter — does it route correctly?
    # ═════════════════════════════════════════════════════════════════════════
    print_section("STEP 4: ChatRouter — Routing Tests")

    router = ChatRouter(groq_api_key=GROQ_API_KEY)

    routing_tests = [
        # (query, has_trace_context, expected_agent)
        ("Why is this trace slow?", True, "single_rca"),
        ("Show me all error logs in this trace", True, "single_rca"),
        ("Is there a retry pattern in this trace?", True, "single_rca"),
        ("What is distributed tracing?", False, "general"),
        ("How does OpenTelemetry work?", False, "general"),
        ("What services are involved in this request?", True, "single_rca"),
    ]

    routing_results = []
    for query, has_trace, expected in routing_tests:
        route = await test_routing(router, query, has_trace)
        match = "✅" if route.agent_type == expected else "❌"
        routing_results.append((query, expected, route.agent_type, match))
        print(f"  {match} '{query[:50]}...'")
        print(f"       Expected: {expected}  →  Got: {route.agent_type}")
        print(f"       Reason: {route.reasoning[:70]}...")
        print()

    passed = sum(1 for _, _, _, m in routing_results if m == "✅")
    print(f"  Router Score: {passed}/{len(routing_results)} correct\n")

    # ═════════════════════════════════════════════════════════════════════════
    # Step 5: Test SingleRCAAgent — end-to-end on ALL 8 traces
    # ═════════════════════════════════════════════════════════════════════════
    print_section("STEP 5: SingleRCAAgent — Full End-to-End (All 8 Traces)")
    print("  Testing each trace with: 'Why is this trace slow?'")
    print("  Fast-path traces skip LLM; others call Groq.\n")

    rca_agent = SingleRCAAgent(groq_api_key=GROQ_API_KEY)
    rca_results = []

    for i, (tid, op, tree, intel_result) in enumerate(trace_trees, 1):
        is_fast = intel_result.fast_path_available
        path_label = "⚡FAST" if is_fast else "🤖 LLM"
        print(f"  ── Trace {i}/{len(trace_trees)}: {op} [{path_label}] ──")

        response, elapsed_ms = await test_rca_agent(
            rca_agent, tid, tree, "Why is this trace slow?"
        )

        # Show first 200 chars of response
        preview = response.message[:200].replace('\n', ' ')
        print(f"     Time: {elapsed_ms:.0f}ms  |  Refs: {len(response.reference)} spans")
        print(f"     Response: {preview}...")

        # Show validation results from synthesizer
        if response.validation_passed is not None:
            v_icon = "✅" if response.validation_passed else "⚠️"
            fb_icon = " [FALLBACK]" if response.fallback_used else ""
            print(f"     Validation: {v_icon} passed={response.validation_passed}, "
                  f"confidence={response.validation_confidence:.2f}{fb_icon}")
            for note in response.validation_notes:
                print(f"       {note}")
        print()

        rca_results.append({
            "root_op": op,
            "fast_path": is_fast,
            "elapsed_ms": elapsed_ms,
            "refs": len(response.reference),
            "response_preview": response.message[:80],
            "full_response": response.message,
            "validation_passed": response.validation_passed,
            "validation_confidence": response.validation_confidence,
            "fallback_used": response.fallback_used,
        })

    # Step 6: Final Summary
    print_section("FINAL SUMMARY — SingleRCAAgent Results")
    print(f"  {'#':<3} {'Root Operation':<30} {'Path':>6} {'Time':>8} {'Refs':>4} {'Valid':>5} {'Conf':>5}  {'FB':>3}")
    print(f"  {'─'*3} {'─'*30} {'─'*6} {'─'*8} {'─'*4} {'─'*5} {'─'*5}  {'─'*3}")
    for i, r in enumerate(rca_results, 1):
        path = "⚡FAST" if r["fast_path"] else "🤖LLM"
        v = "✅" if r["validation_passed"] else "⚠️" if r["validation_passed"] is not None else "?"
        conf = f"{r['validation_confidence']:.2f}" if r["validation_confidence"] is not None else "  -"
        fb = "YES" if r["fallback_used"] else " no"
        print(f"  {i:<3} {r['root_op']:<30} {path:>6} {r['elapsed_ms']:>7.0f}ms {r['refs']:>4} {v:>5} {conf:>5}  {fb:>3}")

    # Check that responses are unique (Bug 1 verification)
    unique_responses = set(r["full_response"] for r in rca_results)
    print(f"\n  Unique responses: {len(unique_responses)}/{len(rca_results)}")
    if len(unique_responses) == len(rca_results):
        print("  ✅ Every trace got a DIFFERENT response — Bug 1 is fixed!")
    else:
        print("  ⚠️  Some traces got the same response — check output above")

    print()


if __name__ == "__main__":
    asyncio.run(main())
