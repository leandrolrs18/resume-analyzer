from dataclasses import dataclass

from fastapi import Request

from app.core.config import Settings
from app.repositories.audit_logs import AuditLogRepository
from app.services.analyzer import ResumeAnalyzerService
from app.services.document_service import DocumentService
from app.services.healthcheck import HealthcheckService
from app.services.llm_service import LlmService
from app.services.ocr_service import OcrService
from app.services.ranking_service import RankingService
from app.services.summarization_service import SummarizationService


@dataclass
class Container:
    analyzer: ResumeAnalyzerService
    audit_logs: AuditLogRepository
    healthcheck: HealthcheckService

    async def shutdown(self) -> None:
        await self.audit_logs.close()


def build_container(settings: Settings) -> Container:
    audit_logs = AuditLogRepository(settings.mongo_uri, settings.mongo_db)
    ocr = OcrService()
    llm = LlmService(settings.hf_model) if settings.use_local_llm else None
    document_service = DocumentService(ocr, settings)
    summarization_service = SummarizationService(llm, settings.summarization_max_new_tokens)
    ranking_service = RankingService(settings.top_k_citations)
    ranking_service.llm = llm
    analyzer = ResumeAnalyzerService(
        document_service=document_service,
        summarization_service=summarization_service,
        ranking_service=ranking_service,
        audit_logs=audit_logs,
        settings=settings,
    )
    healthcheck = HealthcheckService(audit_logs)
    return Container(analyzer=analyzer, audit_logs=audit_logs, healthcheck=healthcheck)


def get_container(request: Request) -> Container:
    return request.app.state.container
