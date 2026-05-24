from datetime import UTC, datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection

from app.models.schemas import AuditLogEntry, AuditLogResponse


class AuditLogRepository:
    def __init__(self, mongo_uri: str, database_name: str):
        self.client = AsyncIOMotorClient(mongo_uri, serverSelectionTimeoutMS=500)
        self.collection: AsyncIOMotorCollection = self.client[database_name]["audit_logs"]

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
        await self.collection.insert_one(payload.model_dump(mode="json"))

    async def get_logs(self, request_id: str) -> AuditLogResponse:
        documents = (
            await self.collection.find({"request_id": request_id})
            .sort("timestamp", 1)
            .to_list(length=200)
        )
        logs = [AuditLogEntry.model_validate(document) for document in documents]
        return AuditLogResponse(request_id=request_id, logs=logs)

    async def ping(self) -> None:
        await self.client.admin.command("ping")

    async def close(self) -> None:
        self.client.close()
