from app.schemas import HealthResponse


class HealthcheckService:
    def __init__(self, audit_logs):
        self.audit_logs = audit_logs

    async def run(self) -> HealthResponse:
        return HealthResponse(status="ok")
