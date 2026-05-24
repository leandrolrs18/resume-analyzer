import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApplicationError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApplicationError)
    async def handle_application_error(_: Request, exc: ApplicationError) -> JSONResponse:
        logger.warning(
            "application_error", extra={"details": exc.details, "error_message": exc.message}
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unexpected_error", extra={"error_type": exc.__class__.__name__})
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
