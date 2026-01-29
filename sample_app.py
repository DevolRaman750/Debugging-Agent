# sample_app.py
import time
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

resource = Resource.create({
    "service.name": "new-ecommerce-backend"
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


def auth_service():
    with tracer.start_as_current_span("auth_service") as span:
        span.add_event("Validating user token", {"level": "INFO"})
        time.sleep(0.1)

        with tracer.start_as_current_span("redis_cache_lookup") as child:
            child.add_event("Checking session in Redis", {"level": "INFO"})
            time.sleep(0.05)


def inventory_service():
    with tracer.start_as_current_span("inventory_service") as span:
        span.add_event("Checking inventory", {"level": "INFO"})

        with tracer.start_as_current_span("inventory_db_query") as child:
            child.add_event("SELECT stock FROM products", {"level": "INFO"})
            time.sleep(0.1)

        with tracer.start_as_current_span("warehouse_api_call") as child:
            child.add_event("Warehouse API slow", {"level": "WARNING"})
            time.sleep(0.3)


def payment_service():
    with tracer.start_as_current_span("payment_service") as span:
        span.add_event("Initiating payment", {"level": "INFO"})

        for retry in range(2):
            with tracer.start_as_current_span(f"payment_gateway_attempt_{retry+1}") as attempt:
                attempt.add_event(
                    f"Calling payment gateway (attempt {retry+1})",
                    {"level": "INFO"}
                )
                time.sleep(0.2)

                if retry == 0:
                    attempt.add_event("Timeout from bank API", {"level": "ERROR"})
                else:
                    attempt.add_event("Payment authorized", {"level": "INFO"})


def cart_service():
    with tracer.start_as_current_span("cart_service") as span:
        span.add_event("Calculating cart total", {"level": "INFO"})

        with tracer.start_as_current_span("pricing_engine") as child:
            child.add_event("Applying discounts", {"level": "INFO"})
            time.sleep(0.05)

        with tracer.start_as_current_span("tax_service") as child:
            child.add_event("Computing tax", {"level": "INFO"})
            time.sleep(0.05)


def checkout_flow():
    with tracer.start_as_current_span("checkout") as span:
        span.add_event("Checkout started", {"level": "INFO"})

        auth_service()
        cart_service()
        inventory_service()
        payment_service()

        span.add_event("Order successfully placed", {"level": "INFO"})


if __name__ == "__main__":
    checkout_flow()
    print("Complex trace with span-attached logs generated. Open Jaeger UI.")
