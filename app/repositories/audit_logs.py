import logging
from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.schemas import AuditLogEntry, AuditLogResponse

logger = logging.getLogger(__name__)


class AuditLogRepository:
    def __init__(self, mongo_uri: str, database_name: str):
        self.enabled = True
        self._in_memory_logs: dict[str, list[dict[str, Any]]] = {}
        try:
            self.client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=3000)
            self.collection: AsyncIOMotorCollection = self.client[database_name]["audit_logs"]
        except Exception as e:
            logger.error(f"Falha crítica ao inicializar o cliente MongoDB: {e}")
            self.enabled = False

    async def save_log(
        self,
        request_id: str,
        user_id: str,
        query: str | None,
        result: dict[str, Any],
        latency_ms: float,
        status: str,
        costs: dict[str, Any] | None = None,
    ) -> None:
        payload = AuditLogEntry(
            request_id=request_id,
            user_id=user_id,
            timestamp=datetime.now(UTC),
            query=query,
            result=result,
            latency_ms=latency_ms,
            costs=costs or {},
            status=status,
        )
        serialized = payload.model_dump(mode="json")
        if request_id not in self._in_memory_logs:
            self._in_memory_logs[request_id] = []
        self._in_memory_logs[request_id].append(serialized)

        if not self.enabled:
            logger.warning("Log de auditoria ignorado: Repositório MongoDB desabilitado.")
            return

        try:
            await self.collection.insert_one(serialized)
        except Exception as e:
            logger.error(f"Erro ao salvar log no MongoDB: {e}")
            self.enabled = False

    async def get_logs(self, request_id: str) -> AuditLogResponse:
        if self.enabled:
            try:
                documents = (
                    await self.collection.find({"request_id": request_id})
                    .sort("timestamp", 1)
                    .to_list(length=200)
                )
                if documents:
                    logs = [AuditLogEntry.model_validate(document) for document in documents]
                    return AuditLogResponse(request_id=request_id, logs=logs)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch audit logs from MongoDB, falling back to memory: {e}"
                )
                self.enabled = False

        mem_logs = self._in_memory_logs.get(request_id, [])
        logs = [AuditLogEntry.model_validate(log) for log in mem_logs]
        return AuditLogResponse(request_id=request_id, logs=logs)

    async def ping(self) -> None:
        if self.enabled:
            await self.client.admin.command("ping")
        else:
            raise ConnectionError("MongoDB está configurado como desabilitado.")

    async def close(self) -> None:
        if self.enabled:
            self.client.close()
