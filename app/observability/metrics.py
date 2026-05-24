from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

REQUESTS_TOTAL = Counter("requests_total", "Total HTTP requests")
REQUEST_LATENCY_SECONDS = Histogram("request_latency_seconds", "HTTP request latency in seconds")
OCR_FAILURES_TOTAL = Counter("ocr_failures_total", "OCR failures")
LLM_FAILURES_TOTAL = Counter("llm_failures_total", "LLM failures")


def setup_metrics(app) -> None:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
