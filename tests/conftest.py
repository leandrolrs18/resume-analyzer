from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.main import create_app
from app.schemas import AnalyzeResponse, AuditLogEntry, AuditLogResponse, HealthResponse
from app.services.dependencies import Container


class FakeAuditLogRepository:
    def __init__(self):
        self.logs: list[dict] = []

    async def save_log(self, **kwargs):
        self.logs.append(kwargs)

    async def get_logs(self, request_id: str) -> AuditLogResponse:
        entries = [
            AuditLogEntry(
                request_id=item["request_id"],
                user_id=item["user_id"],
                timestamp="2026-01-01T00:00:00Z",
                query=item["query"],
                result=item["result"],
                latency_ms=item["latency_ms"],
                costs=item["costs"],
                status=item["status"],
            )
            for item in self.logs
            if item["request_id"] == request_id
        ]
        return AuditLogResponse(request_id=request_id, logs=entries)

    async def ping(self):
        return None

    async def close(self):
        return None


class FakeAnalyzer:
    def __init__(self):
        self.last_query = None

    async def analyze(
        self, request, files, query, language, request_id, user_id
    ) -> AnalyzeResponse:
        self.last_query = query
        if query:
            return AnalyzeResponse(
                request_id=request_id,
                query=query,
                results=[
                    {
                        "rank": 1,
                        "candidate": "Maria",
                        "score": 0.92,
                        "summary": "Senior backend engineer with Python and AWS background.",
                        "justification": (
                            "Strong overlap with Python and AWS from the provided evidence."
                        ),
                        "citations": [
                            {"chunk_id": "Maria-0", "text": "5 years with AWS and Python"}
                        ],
                    }
                ],
            )
        return AnalyzeResponse(
            request_id=request_id,
            results=[{"candidate": "Maria", "summary": "Experienced backend engineer."}],
        )


class FakeHealthcheck:
    async def run(self) -> HealthResponse:
        return HealthResponse(status="ok")


@pytest.fixture
def app() -> FastAPI:
    app = create_app()
    app.state.container = Container(
        analyzer=FakeAnalyzer(),
        audit_logs=FakeAuditLogRepository(),
        healthcheck=FakeHealthcheck(),
    )
    return app


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
