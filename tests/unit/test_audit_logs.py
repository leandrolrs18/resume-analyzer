import pytest

from app.repositories.audit_logs import AuditLogRepository


class FailingCollection:
    async def insert_one(self, payload):
        del payload
        raise ConnectionError("mongo unavailable")


# Garante que a auditoria continua disponível em memória quando o Mongo falha ao salvar.
@pytest.mark.asyncio
async def test_audit_log_falls_back_to_memory_when_mongo_save_fails() -> None:
    repository = AuditLogRepository("mongodb://localhost:27017", "resume-analyzer-test")
    repository.collection = FailingCollection()

    await repository.save_log(
        request_id="req-1",
        user_id="user-1",
        query="python",
        result={"request_id": "req-1", "results": []},
        latency_ms=10.5,
        status="success",
        costs={"llm_provider": "gemini"},
    )

    response = await repository.get_logs("req-1")

    assert response.request_id == "req-1"
    assert len(response.logs) == 1
    assert response.logs[0].request_id == "req-1"
    assert response.logs[0].result["request_id"] == "req-1"
