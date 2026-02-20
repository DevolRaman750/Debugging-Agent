# main.py
# Test script for Intelligence Layer - processes trace and outputs IntelligenceResult
import asyncio
import json
from datetime import datetime

from src.config import GROQ_API_KEY
from src.service.provider import ObservabilityProvider
from src.context.tree_builder import build_heterogeneous_tree
from src.context.utils import find_root_span
from src.routing.types import ChatMode, ChatModel
from src.routing.router import ChatRouter
from src.agents.single_rca_agent import SingleRCAAgent
from src.intel.pipeline import IntelligencePipeline


def print_section(title: str, width: int = 60):
    """Helper to print formatted section headers."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


async def main():
    # =========================================================================
    # STEP 1: Fetch Trace from Jaeger
    # =========================================================================
    print_section("STEP 1: Fetching Trace from Jaeger")
    
    provider = ObservabilityProvider.create_jaeger_provider()
    
    # Replace with your trace_id from Jaeger UI
    trace_id = "aab7900d7a8d59d04eebed386ed6924e"
    
    print(f"Trace ID: {trace_id}")
    
    trace = await provider.trace_client.get_trace_by_id(trace_id)
    
    if not trace:
        print(f"❌ Trace {trace_id} not found")
        return
    
    print(f"✅ Trace fetched: {len(trace.spans)} root spans found")
    
    # =========================================================================
    # STEP 2: Fetch Logs
    # =========================================================================
    print_section("STEP 2: Fetching Logs")
    
    logs = await provider.log_client.get_logs_by_trace_id(trace_id)
    print(f"✅ Logs fetched: {len(logs.logs)} span log groups")
    
    # =========================================================================
    # STEP 3: Build Heterogeneous Tree
    # =========================================================================
    print_section("STEP 3: Building Span Tree")
    
    root_span = find_root_span(trace.spans)
    tree = build_heterogeneous_tree(root_span, logs.logs)
    
    print(f"✅ Tree built with root span: {tree.func_full_name}")
    print(f"   - Latency: {tree.span_latency:.2f}ms")
    print(f"   - Children: {len(tree.children_spans)} spans")
    
    # =========================================================================
    # STEP 4: Run Intelligence Pipeline
    # =========================================================================
    print_section("STEP 4: Running Intelligence Pipeline")
    
    intel_pipeline = IntelligencePipeline()
    intel_result = await intel_pipeline.process(
        tree=tree,
        user_query="Why is this trace slow?"
    )
    
    print(f"✅ Intelligence processing completed in {intel_result.processing_time_ms:.2f}ms")
    
    # =========================================================================
    # STEP 5: Display Results
    # =========================================================================
    
    # 5.1 Classification Results
    print_section("5.1 Classification Results")
    print(f"Root span type: {intel_result.classified_tree.span_type.value}")
    print(f"Classification confidence: {intel_result.classified_tree.classification_confidence:.2f}")
    
    # Print all classified spans so you can verify different traces → different data
    def print_classified_tree(node, indent=0):
        prefix = "  " * indent
        print(f"{prefix}[{node.span_type.value}] {node.func_full_name} "
              f"({node.span_latency:.1f}ms, conf={node.classification_confidence:.2f}, "
              f"logs={len(node.logs)})")
        for child in node.children_spans:
            print_classified_tree(child, indent + 1)
    
    print("\nFull span tree:")
    print_classified_tree(intel_result.classified_tree)
    
    # 5.2 Suppression Stats
    print_section("5.2 Suppression Stats")
    stats = intel_result.suppression_stats
    print(f"Original spans: {stats.original_span_count}")
    print(f"Remaining spans: {stats.remaining_span_count}")
    print(f"Suppressed: {stats.suppressed_count}")
    print(f"Compression ratio: {stats.compression_ratio:.2f}x")
    
    # 5.3 Failure Report
    print_section("5.3 Failure Report")
    failures = intel_result.failure_report
    print(f"Has failures: {failures.has_failures}")
    print(f"Failure spans found: {len(failures.failure_spans)}")
    
    if failures.failure_spans:
        print("\nFailure details:")
        for fs in failures.failure_spans:
            print(f"  - [{fs.failure_type.value}] {fs.span_function}")
            print(f"    Root cause: {fs.is_root_cause}")
            if fs.error_messages:
                print(f"    Error: {fs.error_messages[0][:50]}...")
    
    # 5.4 Pattern Matches
    print_section("5.4 Pattern Matches")
    if intel_result.pattern_matches:
        for pm in intel_result.pattern_matches:
            print(f"  🔍 {pm.pattern_name} (confidence: {pm.confidence:.2f})")
            print(f"     Category: {pm.pattern_category}")
            print(f"     Explanation: {pm.explanation[:80]}...")
            print(f"     Fix: {pm.recommended_fix[:80]}...")
    else:
        print("  No known patterns matched")
    
    # 5.5 Ranked Causes
    print_section("5.5 Ranked Causes")
    ranked = intel_result.ranked_causes
    print(f"Confidence level: {ranked.confidence_level.value}")
    print(f"Top cause score: {ranked.top_cause_score:.2f}")
    print(f"Score gap: {ranked.score_gap:.2f}")
    
    if ranked.causes:
        print("\nTop causes:")
        for cause in ranked.causes[:3]:
            print(f"  #{cause.rank} {cause.span_function} (score: {cause.score:.2f})")
    
    # 5.6 Compression Stats
    print_section("5.6 Context Compression")
    compressed = intel_result.compressed_context
    print(f"Original tokens: {compressed.original_token_count}")
    print(f"Compressed tokens: {compressed.token_count}")
    print(f"Compression ratio: {compressed.compression_ratio:.2f}x")
    
    # 5.7 Fast Path
    print_section("5.7 Fast Path Availability")
    if intel_result.fast_path_available:
        print("✅ Fast path available - can skip LLM reasoning!")
        print(f"   Use pattern explanation: {intel_result.pattern_matches[0].explanation}")
    else:
        print("❌ Fast path not available - full LLM reasoning needed")
    
    # =========================================================================
    # STEP 6: Export Full Result as JSON
    # =========================================================================
    print_section("STEP 6: Full Intelligence Result (JSON)")
    
    # Convert to dict for JSON serialization
    result_dict = intel_result.model_dump(mode='json')
    
    # Save to file
    output_file = "intelligence_result.json"
    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    
    print(f"✅ Full result saved to: {output_file}")
    print("\nPreview (first 500 chars):")
    print(json.dumps(result_dict, indent=2, default=str)[:500] + "...")

    # =========================================================================
    # STEP 7: Test ChatRouter (Agent Routing)
    # =========================================================================
    print_section("STEP 7: Testing ChatRouter")
    
    router = ChatRouter(groq_api_key=GROQ_API_KEY)
    
    # Test 1: Should route to "single_rca" (has trace + asking about issue)
    route_result = await router.route_query(
        user_message="Why is this trace slow?",
        chat_mode=ChatMode.AGENT,
        has_trace_context=True,
    )
    print(f"  Query: 'Why is this trace slow?'")
    print(f"  ✅ Routed to: {route_result.agent_type}")
    print(f"     Reasoning: {route_result.reasoning}")
    
    # Test 2: Should route to "general" (no trace, general question)
    route_result_2 = await router.route_query(
        user_message="What is distributed tracing?",
        chat_mode=ChatMode.CHAT,
        has_trace_context=False,
    )
    print(f"\n  Query: 'What is distributed tracing?'")
    print(f"  ✅ Routed to: {route_result_2.agent_type}")
    print(f"     Reasoning: {route_result_2.reasoning}")
    
    # =========================================================================
    # STEP 8: Test SingleRCAAgent (Full End-to-End)
    # =========================================================================
    print_section("STEP 8: Testing SingleRCAAgent (Full Pipeline)")
    print("  This will: Intel Layer → Feature Select → Build Context → Groq LLM")
    print("  Please wait...\n")
    
    rca_agent = SingleRCAAgent(groq_api_key=GROQ_API_KEY)
    
    rca_response = await rca_agent.chat(
        trace_id=trace_id,
        chat_id="test-chat-001",
        user_message="Why is this trace slow?",
        model=ChatModel.AUTO,
        timestamp=datetime.now(),
        tree=tree,
        chat_history=None,
    )
    
    print(f"  ✅ RCA Response received!")
    print(f"  References: {len(rca_response.reference)} spans cited")
    print(f"\n{'─' * 60}")
    print(f"  RESPONSE:")
    print(f"{'─' * 60}")
    print(rca_response.message)
    
    # =========================================================================
    # STEP 9: Test with Different Queries
    # =========================================================================
    print_section("STEP 9: Testing Different User Queries")
    
    test_queries = [
        "Show me all error logs in this trace",
        "What services are involved in this request?",
        "Is there a retry pattern in this trace?",
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        
        route = await router.route_query(
            user_message=query,
            chat_mode=ChatMode.AGENT,
            has_trace_context=True,
        )
        print(f"  → Routed to: {route.agent_type} ({route.reasoning[:50]}...)")


if __name__ == "__main__":
    asyncio.run(main())
