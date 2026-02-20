# sample_app.py
# ═══════════════════════════════════════════════════════════════════════════════
# Enhanced Sample App - Generates 8 diverse traces for testing the FULL pipeline
# (Intelligence Layer → Routing Layer → LLM Reasoning)
#
# Run: python -m src.sample_app
# View: http://localhost:16686 (Jaeger UI)
# ═══════════════════════════════════════════════════════════════════════════════

import time
import random
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "ecommerce-backend-2"
})

trace.set_tracer_provider(TracerProvider(resource=resource))

otlp_exporter = OTLPSpanExporter(
    endpoint="http://localhost:4317",
    insecure=True
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

tracer = trace.get_tracer(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1: Slow Database Query
# Tests: SlowQueryPattern, latency anomaly detection, feature selection
# Expected: Pattern match with "slow_query", confidence ≥ 0.8
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_1_slow_db_query():
    """Simulates a checkout where the product lookup query takes 2.5 seconds."""
    with tracer.start_as_current_span("POST /api/checkout") as root:
        root.add_event("Checkout request received", {"level": "INFO"})

        # Auth - fast
        with tracer.start_as_current_span("auth_service") as auth:
            auth.add_event("Token validated", {"level": "INFO"})
            time.sleep(0.05)

        # Cart lookup - fast
        with tracer.start_as_current_span("cart_service") as cart:
            cart.add_event("Cart loaded for user_123", {"level": "INFO"})
            time.sleep(0.03)

        # ══ THE SLOW QUERY ══
        with tracer.start_as_current_span("product_db_query") as db:
            db.add_event("SELECT * FROM products JOIN inventory ON ... WHERE cart_items IN (...)", {"level": "INFO"})
            db.add_event("Query execution plan: FULL TABLE SCAN on products", {"level": "WARNING"})
            time.sleep(2.5)  # 2500ms - very slow!
            db.add_event("Query returned 47 rows", {"level": "INFO"})

        # Payment - normal speed
        with tracer.start_as_current_span("payment_gateway") as pay:
            pay.add_event("Charging card ending in 4242", {"level": "INFO"})
            time.sleep(0.2)
            pay.add_event("Payment authorized", {"level": "INFO"})

        root.add_event("Checkout completed", {"level": "INFO"})
    print("  ✅ Scenario 1: Slow DB Query (2.5s) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2: N+1 Query Anti-Pattern
# Tests: N1QueryPattern, span grouping by parent
# Expected: Pattern match with "n_plus_1_query", 6 similar queries detected
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_2_n_plus_1_query():
    """Simulates loading an order page that makes individual DB calls per item."""
    with tracer.start_as_current_span("GET /api/orders/456") as root:
        root.add_event("Loading order details", {"level": "INFO"})

        # First: get the order
        with tracer.start_as_current_span("db_get_order") as db:
            db.add_event("SELECT * FROM orders WHERE id = 456", {"level": "INFO"})
            time.sleep(0.02)

        # Then: N+1 — individual queries for each line item
        with tracer.start_as_current_span("load_order_items") as loader:
            loader.add_event("Loading 6 order items individually", {"level": "INFO"})

            for i in range(6):
                with tracer.start_as_current_span(f"db_get_product_{i+1}") as item:
                    item.add_event(
                        f"SELECT * FROM products WHERE id = {100 + i}",
                        {"level": "INFO"}
                    )
                    time.sleep(0.03)  # 30ms each × 6 = 180ms total (should be 1 query)

        # Get shipping info
        with tracer.start_as_current_span("shipping_service") as ship:
            ship.add_event("Calculating shipping estimate", {"level": "INFO"})
            time.sleep(0.05)

        root.add_event("Order page loaded", {"level": "INFO"})
    print("  ✅ Scenario 2: N+1 Query (6 individual queries) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3: Retry Storm with Errors
# Tests: RetryPattern, error log detection, failure classification
# Expected: 4 failed attempts + 1 success, ERROR logs detected
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_3_retry_storm():
    """Simulates payment service retrying 4 times before success."""
    with tracer.start_as_current_span("POST /api/payment") as root:
        root.add_event("Payment request received for $89.99", {"level": "INFO"})

        with tracer.start_as_current_span("payment_processor") as processor:
            processor.add_event("Starting payment processing", {"level": "INFO"})

            for attempt in range(5):
                with tracer.start_as_current_span(f"stripe_api_call_attempt_{attempt+1}") as retry:
                    retry.add_event(
                        f"Calling Stripe API (attempt {attempt+1}/5)",
                        {"level": "INFO"}
                    )
                    time.sleep(0.3)

                    if attempt < 4:
                        # First 4 attempts fail
                        error_msgs = [
                            "Connection reset by peer",
                            "Gateway timeout (504)",
                            "Rate limit exceeded (429)",
                            "Internal server error (500)",
                        ]
                        retry.add_event(
                            f"FAILED: {error_msgs[attempt]}",
                            {"level": "ERROR"}
                        )
                        retry.add_event(
                            f"Waiting {2**attempt}s before retry (exponential backoff)",
                            {"level": "WARNING"}
                        )
                        time.sleep(0.1 * (attempt + 1))  # Simulated backoff
                    else:
                        # 5th attempt succeeds
                        retry.add_event("Payment authorized: txn_abc123", {"level": "INFO"})

            processor.add_event("Payment completed after 5 attempts", {"level": "WARNING"})

        root.add_event("Payment processed successfully", {"level": "INFO"})
    print("  ✅ Scenario 3: Retry Storm (4 failures + 1 success) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4: Cascading Timeout
# Tests: Failure propagation, root cause vs symptom ranking
# Expected: Root cause = database timeout (deepest), not the API timeout (surface)
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_4_cascading_timeout():
    """Simulates: API → Service → Database, where DB timeout cascades up."""
    with tracer.start_as_current_span("GET /api/search") as api:
        api.add_event("Search request: query='laptop'", {"level": "INFO"})

        with tracer.start_as_current_span("search_service") as service:
            service.add_event("Processing search query", {"level": "INFO"})

            with tracer.start_as_current_span("elasticsearch_query") as es:
                es.add_event("Querying Elasticsearch cluster", {"level": "INFO"})

                # ══ ROOT CAUSE: ES connection timeout ══
                with tracer.start_as_current_span("es_node_connection") as node:
                    node.add_event("Connecting to es-node-3:9200", {"level": "INFO"})
                    time.sleep(3.0)  # 3s timeout — the actual root cause
                    node.add_event("Connection timed out to es-node-3", {"level": "ERROR"})

                es.add_event("Elasticsearch query failed: connection timeout", {"level": "ERROR"})
                time.sleep(0.01)

            # Service tries a fallback
            with tracer.start_as_current_span("fallback_cache_lookup") as cache:
                cache.add_event("Trying Redis cache fallback", {"level": "WARNING"})
                time.sleep(0.05)
                cache.add_event("Cache miss for query 'laptop'", {"level": "WARNING"})

            service.add_event("Search service failed: no results available", {"level": "ERROR"})

        api.add_event("Search request failed: 504 Gateway Timeout", {"level": "ERROR"})
    print("  ✅ Scenario 4: Cascading Timeout (DB → Service → API) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 5: Health Check Noise
# Tests: Suppressor (should suppress health checks), tree pruning
# Expected: 10 health checks suppressed, 1 real request remains
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_5_health_check_noise():
    """Simulates a trace with 10 health checks and 1 real request."""
    with tracer.start_as_current_span("request_handler") as root:
        root.add_event("Processing incoming requests", {"level": "INFO"})

        # 10 health check spans — should be suppressed
        for i in range(10):
            with tracer.start_as_current_span("health_check") as hc:
                hc.add_event("Health check ping", {"level": "INFO"})
                time.sleep(0.001)  # 1ms each — very fast

        # 1 real request with actual work
        with tracer.start_as_current_span("POST /api/orders") as order:
            order.add_event("Creating new order", {"level": "INFO"})

            with tracer.start_as_current_span("validate_input") as validate:
                validate.add_event("Validating order payload", {"level": "INFO"})
                time.sleep(0.02)

            with tracer.start_as_current_span("save_to_db") as save:
                save.add_event("INSERT INTO orders (...)", {"level": "INFO"})
                time.sleep(0.1)
                save.add_event("Order #789 saved", {"level": "INFO"})

            with tracer.start_as_current_span("send_confirmation_email") as email:
                email.add_event("Sending email to user@example.com", {"level": "INFO"})
                time.sleep(0.05)

        root.add_event("All requests processed", {"level": "INFO"})
    print("  ✅ Scenario 5: Health Check Noise (10 health + 1 real) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 6: Mixed Success and Failure
# Tests: Classifier accuracy, selective failure detection
# Expected: Only the 2 ERROR spans detected, 4 OK spans ignored
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_6_mixed_errors():
    """Simulates a request where some services succeed and others fail."""
    with tracer.start_as_current_span("POST /api/submit-report") as root:
        root.add_event("Report submission started", {"level": "INFO"})

        # ✅ Success: Upload file
        with tracer.start_as_current_span("file_upload_service") as upload:
            upload.add_event("Uploading report.pdf (2.3MB)", {"level": "INFO"})
            time.sleep(0.15)
            upload.add_event("File uploaded to S3: s3://reports/report.pdf", {"level": "INFO"})

        # ❌ Failure: Virus scan
        with tracer.start_as_current_span("virus_scan_service") as scan:
            scan.add_event("Scanning report.pdf for malware", {"level": "INFO"})
            time.sleep(0.3)
            scan.add_event("Virus scan service unavailable: connection refused", {"level": "ERROR"})

        # ✅ Success: Metadata extraction
        with tracer.start_as_current_span("metadata_service") as meta:
            meta.add_event("Extracting document metadata", {"level": "INFO"})
            time.sleep(0.05)
            meta.add_event("Metadata extracted: 12 pages, PDF 2.0", {"level": "INFO"})

        # ❌ Failure: Notification
        with tracer.start_as_current_span("notification_service") as notify:
            notify.add_event("Sending Slack notification", {"level": "INFO"})
            time.sleep(0.1)
            notify.add_event("Slack webhook returned 403: invalid token", {"level": "ERROR"})

        # ✅ Success: Save record
        with tracer.start_as_current_span("db_save_report") as save:
            save.add_event("INSERT INTO reports (...)", {"level": "INFO"})
            time.sleep(0.04)
            save.add_event("Report record saved", {"level": "INFO"})

        # ✅ Success: Audit log
        with tracer.start_as_current_span("audit_log") as audit:
            audit.add_event("Audit log written for report submission", {"level": "INFO"})
            time.sleep(0.01)

        root.add_event("Report submitted with partial failures", {"level": "WARNING"})
    print("  ✅ Scenario 6: Mixed Success/Failure generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 7: Deep Span Tree (15 levels)
# Tests: Tree building depth, context chunking for large trees
# Expected: Tree correctly built, context may need chunking
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_7_deep_tree():
    """Simulates a deeply nested call chain: API → Gateway → Auth → ... → DB."""
    service_names = [
        "api_gateway", "load_balancer", "auth_middleware", "rate_limiter",
        "request_validator", "business_logic", "data_transformer",
        "cache_check", "primary_db_pool", "connection_manager",
        "query_optimizer", "index_lookup", "table_scan", "row_fetch",
        "result_serializer"
    ]

    def create_nested_spans(depth, names):
        if depth >= len(names):
            return
        with tracer.start_as_current_span(names[depth]) as span:
            span.add_event(f"Processing at level {depth + 1}", {"level": "INFO"})
            time.sleep(0.02)
            if depth == 12:  # Add an error deep in the stack
                span.add_event("Table lock contention detected", {"level": "ERROR"})
                time.sleep(0.5)
            create_nested_spans(depth + 1, names)

    with tracer.start_as_current_span("GET /api/user/profile") as root:
        root.add_event("Deep nested request", {"level": "INFO"})
        create_nested_spans(0, service_names)
        root.add_event("Request completed", {"level": "INFO"})

    print("  ✅ Scenario 7: Deep Tree (15 levels) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO 8: Happy Path (No Errors)
# Tests: General agent routing, "no issues found" response
# Expected: No patterns match, no failures, general agent response
# ═══════════════════════════════════════════════════════════════════════════════

def scenario_8_happy_path():
    """Simulates a perfectly working request — everything fast, no errors."""
    with tracer.start_as_current_span("GET /api/dashboard") as root:
        root.add_event("Dashboard request", {"level": "INFO"})

        with tracer.start_as_current_span("user_profile_cache") as cache:
            cache.add_event("Cache HIT for user_123", {"level": "INFO"})
            time.sleep(0.005)

        with tracer.start_as_current_span("recent_orders_query") as orders:
            orders.add_event("SELECT TOP 5 FROM orders WHERE user_id = 123", {"level": "INFO"})
            time.sleep(0.025)
            orders.add_event("5 orders returned", {"level": "INFO"})

        with tracer.start_as_current_span("recommendations_service") as recs:
            recs.add_event("Fetching personalized recommendations", {"level": "INFO"})
            time.sleep(0.03)
            recs.add_event("8 recommendations generated", {"level": "INFO"})

        with tracer.start_as_current_span("render_dashboard") as render:
            render.add_event("Rendering dashboard HTML", {"level": "INFO"})
            time.sleep(0.01)

        root.add_event("Dashboard served in 70ms", {"level": "INFO"})
    print("  ✅ Scenario 8: Happy Path (no errors) generated")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  TraceRoot Test Trace Generator")
    print("  Generating 8 test scenarios...")
    print("=" * 65)
    print()

    scenario_1_slow_db_query()
    scenario_2_n_plus_1_query()
    scenario_3_retry_storm()
    scenario_4_cascading_timeout()
    scenario_5_health_check_noise()
    scenario_6_mixed_errors()
    scenario_7_deep_tree()
    scenario_8_happy_path()

    # Give BatchSpanProcessor time to flush
    time.sleep(2)

    print()
    print("=" * 65)
    print("  All 8 traces generated!")
    print("  View them at: http://localhost:16686")
    print("  Service name: ecommerce-backend")
    print()
    print("  Copy each trace_id from Jaeger → paste into test scripts")
    print("=" * 65)
