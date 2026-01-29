import asyncio
import json
from src.service.provider import ObservabilityProvider
from src.context.tree_builder import build_heterogeneous_tree
from src.context.utils import find_root_span
async def main():
    # Create provider
    provider = ObservabilityProvider.create_jaeger_provider()
    
    # Fetch trace (replace with real trace_id from Jaeger UI)
    trace_id = "d199baed84108ce47c0ca422a78db615"
    trace = await provider.trace_client.get_trace_by_id(trace_id)
    
    if not trace:
        print(f"Trace {trace_id} not found")
        return
    
    # Fetch logs
    logs = await provider.log_client.get_logs_by_trace_id(trace_id)

    root_span = find_root_span(trace.spans)
    
    # Build heterogeneous tree
    tree = build_heterogeneous_tree(root_span, logs.logs)
    
    # Print as JSON (this is what AI would see)
    print(json.dumps(tree.to_dict(), indent=2))
if __name__ == "__main__":
    asyncio.run(main())