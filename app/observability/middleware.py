import logging
import time
from contextvars import ContextVar
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.observability.metrics import REQUEST_LATENCY_SECONDS, REQUESTS_TOTAL

request_context: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid4())
        request.state.request_id = request_id
        token = request_context.set(request_id)
        REQUESTS_TOTAL.inc()
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed = time.perf_counter() - started_at
            REQUEST_LATENCY_SECONDS.observe(elapsed)
            logger.info(
                "request_complete",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "latency_ms": round(elapsed * 1000, 2),
                },
            )
            request_context.reset(token)
        response.headers["x-request-id"] = request_id
        return response
