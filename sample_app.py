# sample_app.py
# Enhanced sample app to generate diverse traces for testing Intelligence Layer
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "ecommerce-backend"
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


# ============================================================================
# SERVICE FUNCTIONS - Simulating real microservices
# ============================================================================

def auth_service():
    """Simulates authentication service with Redis cache lookup."""
    with tracer.start_as_current_span("auth_service") as span:
        span.add_event("Validating user token", {"level": "INFO"})
        time.sleep(0.1)

        with tracer.start_as_current_span("redis_cache_lookup") as child:
            child.add_event("Checking session in Redis cache", {"level": "INFO"})
            time.sleep(0.05)


def inventory_service():
    """Simulates inventory check with DB query and external API call."""
    with tracer.start_as_current_span("inventory_service") as span:
        span.add_event("Checking inventory levels", {"level": "INFO"})

        # Database query - will be classified as DB_QUERY
        with tracer.start_as_current_span("inventory_db_query") as child:
            child.add_event("SELECT stock FROM products WHERE id = ?", {"level": "INFO"})
            time.sleep(0.1)

        # Slow external API - will trigger latency anomaly
        with tracer.start_as_current_span("warehouse_api_call") as child:
            child.add_event("Calling warehouse API", {"level": "INFO"})
            child.add_event("Warehouse API response slow", {"level": "WARNING"})
            time.sleep(0.3)


def payment_service():
    """Simulates payment with retry pattern and timeout error."""
    with tracer.start_as_current_span("payment_service") as span:
        span.add_event("Initiating payment flow", {"level": "INFO"})

        # Retry pattern - will be detected
        for retry in range(2):
            with tracer.start_as_current_span(f"payment_gateway_attempt_{retry+1}") as attempt:
                attempt.add_event(
                    f"Calling payment gateway (attempt {retry+1})",
                    {"level": "INFO"}
                )
                time.sleep(0.2)

                if retry == 0:
                    # First attempt fails with timeout - ERROR log
                    attempt.add_event("Timeout from bank API", {"level": "ERROR"})
                else:
                    attempt.add_event("Payment authorized successfully", {"level": "INFO"})


def cart_service():
    """Simulates cart operations with pricing and tax calculations."""
    with tracer.start_as_current_span("cart_service") as span:
        span.add_event("Calculating cart total", {"level": "INFO"})

        with tracer.start_as_current_span("pricing_engine") as child:
            child.add_event("Applying discount codes", {"level": "INFO"})
            time.sleep(0.05)

        with tracer.start_as_current_span("tax_service") as child:
            child.add_event("Computing tax for region", {"level": "INFO"})
            time.sleep(0.05)


def notification_service():
    """Simulates notification with queue publish."""
    with tracer.start_as_current_span("notification_service") as span:
        span.add_event("Preparing order confirmation", {"level": "INFO"})
        
        with tracer.start_as_current_span("kafka_publish_order_event") as child:
            child.add_event("Publishing to kafka topic: order-events", {"level": "INFO"})
            time.sleep(0.02)


def n_plus_1_query_example():
    """Simulates N+1 query anti-pattern for pattern detection testing."""
    with tracer.start_as_current_span("user_loader") as span:
        span.add_event("Loading users with orders", {"level": "INFO"})
        
        # This creates N+1 pattern - multiple similar DB queries
        for i in range(5):
            with tracer.start_as_current_span(f"db_query_user_{i}") as child:
                child.add_event(f"SELECT * FROM orders WHERE user_id = {i}", {"level": "INFO"})
                time.sleep(0.02)


def health_check():
    """Health check span - should be suppressed by Intelligence Layer."""
    with tracer.start_as_current_span("health_check") as span:
        span.add_event("Health check ping", {"level": "INFO"})
        time.sleep(0.001)


# ============================================================================
# MAIN CHECKOUT FLOW
# ============================================================================

def checkout_flow():
    """Main checkout flow that exercises all services."""
    with tracer.start_as_current_span("checkout") as span:
        span.add_event("Checkout flow started", {"level": "INFO"})

        # Health check (should be suppressed)
        health_check()

        # Core services
        auth_service()
        cart_service()
        inventory_service()
        
        # N+1 pattern example
        n_plus_1_query_example()
        
        # Payment with retry/timeout
        payment_service()
        
        # Notification
        notification_service()

        span.add_event("Order successfully placed", {"level": "INFO"})


if __name__ == "__main__":
    checkout_flow()
    print("=" * 60)
    print("Trace generated! Check Jaeger UI at http://localhost:16686")
    print("Copy the trace_id and paste it into main.py to test Intelligence Layer")
    print("=" * 60)
