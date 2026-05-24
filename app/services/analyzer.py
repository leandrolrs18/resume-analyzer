import asyncio
import logging
import time
from typing import Any

from fastapi import Request, UploadFile

from app.core.config import Settings
from app.core.security import validate_uploads
from app.rag.chunker import TextChunker
from app.repositories.audit_logs import AuditLogRepository
from app.schemas import AnalyzeResponse, RankingResult, SummaryResult
from app.services.document_service import DocumentService
from app.services.ranking_service import RankingService
from app.services.summarization_service import SummarizationService

logger = logging.getLogger(__name__)


class ResumeAnalyzerService:
    def __init__(
        self,
        document_service: DocumentService,
        summarization_service: SummarizationService,
        ranking_service: RankingService,
        audit_logs: AuditLogRepository,
        settings: Settings,
    ):
        self.document_service = document_service
        self.summarization_service = summarization_service
        self.ranking_service = ranking_service
        self.audit_logs = audit_logs
        self.settings = settings
        self.chunker = TextChunker(settings.chunk_size, settings.chunk_overlap)

    async def analyze(
        self,
        request: Request,
        files: list[UploadFile],
        query: str | None,
        language: str,
        request_id: str,
        user_id: str,
    ) -> AnalyzeResponse:
        validate_uploads(files, self.settings)
        started_at = time.perf_counter()
        status = "success"
        response_payload: dict[str, Any] = {}
        try:
            stage_started = time.perf_counter()
            documents = await self.document_service.extract_documents(files)
            self._log_stage("documents_extracted", stage_started, request_id)

            stage_started = time.perf_counter()
            summaries = await asyncio.gather(
                *(self.summarization_service.summarize(document) for document in documents)
            )
            for document, summary in zip(documents, summaries, strict=False):
                document.summary = summary
                document.chunks = self.chunker.split(document.candidate, document.extracted_text)
            self._log_stage("documents_chunked_and_summarized", stage_started, request_id)

            if not query:
                results = [
                    SummaryResult(candidate=document.candidate, summary=document.summary or "")
                    for document in documents
                ]
                response = AnalyzeResponse(request_id=request_id, results=results)
            else:
                stage_started = time.perf_counter()
                evidence = await self.ranking_service.rank(query, documents)
                self._log_stage("in_memory_ranking_completed", stage_started, request_id)

                stage_started = time.perf_counter()
                documents_by_candidate = {document.candidate: document for document in documents}
                synthesized = await self.summarization_service.synthesize_ranked_results(
                    query=query,
                    language=language,
                    evidence=evidence,
                    documents_by_candidate=documents_by_candidate,
                    max_new_tokens=self._synthesis_max_tokens(),
                )
                self._log_stage("single_llm_synthesis_completed", stage_started, request_id)

                results: list[RankingResult] = []
                for rank, item in enumerate(evidence, start=1):
                    synthesized_item = synthesized.get(item.candidate, {})
                    results.append(
                        RankingResult(
                            rank=rank,
                            candidate=item.candidate,
                            score=round(item.score, 4),
                            summary=synthesized_item.get(
                                "summary", documents_by_candidate[item.candidate].summary or ""
                            ),
                            justification=synthesized_item.get("justification", ""),
                            citations=item.citations,
                        )
                    )
                response = AnalyzeResponse(request_id=request_id, query=query, results=results)

            response_payload = response.model_dump(mode="json")
            return response
        except Exception:
            status = "failure"
            raise
        finally:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            try:
                await self.audit_logs.save_log(
                    request_id=request_id,
                    user_id=user_id,
                    query=query,
                    result=response_payload,
                    latency_ms=latency_ms,
                    status=status,
                    costs={
                        "model": "Qwen2.5-0.5B-Instruct-GGUF",
                        "ranking": "in_memory_bm25",
                        "ocr": "tesseract",
                    },
                )
            except Exception:
                logger.exception("audit_log_save_failed", extra={"request_id": request_id})
            logger.info(
                "analysis_completed",
                extra={
                    "request_id": request_id,
                    "user_id": user_id,
                    "query": query,
                    "status": status,
                    "latency_ms": latency_ms,
                    "path": request.url.path,
                },
            )

    @staticmethod
    def _log_stage(stage: str, started_at: float, request_id: str) -> None:
        logger.info(
            "analysis_stage_completed",
            extra={
                "request_id": request_id,
                "stage": stage,
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )

    def _synthesis_max_tokens(self) -> int:
        configured = (
            self.settings.summarization_max_new_tokens + self.settings.justification_max_new_tokens
        )
        return min(160, configured)
