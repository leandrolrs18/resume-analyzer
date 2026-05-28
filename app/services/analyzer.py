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
        self._document_cache: dict[str, Any] = {}

    async def analyze(
        self,
        request: Request,
        files: list[UploadFile],
        query: str | None,
        language: str,
        llm_provider: str,
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
            self._prune_document_cache(
                {document.cache_key for document in documents if document.cache_key}
            )
            self._log_stage("documents_extracted", stage_started, request_id)

            stage_started = time.perf_counter()
            documents = await self._prepare_documents(documents, llm_provider, request_id)
            self._log_stage("documents_structured_and_chunked", stage_started, request_id)

            if not query:
                stage_started = time.perf_counter()
                await self._summarize_documents(documents, language, llm_provider)
                self._log_stage("documents_summarized", stage_started, request_id)
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
                    llm_provider=llm_provider,
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
                            summary=synthesized_item.get("summary", ""),
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
                        "llm_provider": llm_provider,
                        "retrieval_strategy": "hybrid",
                        "ranking": "in_memory_hybrid_bm25_embeddings_rerank",
                        "parsing": "section_splitter_spacy_json_in_memory",
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
        return min(100, max(80, configured))

    async def _prepare_documents(
        self,
        documents: list,
        llm_provider: str,
        request_id: str,
    ) -> list:
        del llm_provider, request_id
        prepared = []
        to_process = []
        for index, document in enumerate(documents):
            cache_key = document.cache_key
            if cache_key and cache_key in self._document_cache:
                cached = self._document_cache[cache_key].model_copy(deep=True)
                prepared.append((index, cached))
            else:
                to_process.append((index, document))

        for index, document in to_process:
            raw_chunks = self.chunker.split(document.candidate, document.extracted_text)
            document.chunks = raw_chunks
            if document.cache_key:
                self._document_cache[document.cache_key] = document.model_copy(deep=True)
            prepared.append((index, document))

        return [document for _, document in sorted(prepared, key=lambda item: item[0])]

    async def _summarize_documents(
        self,
        documents: list,
        language: str,
        llm_provider: str,
    ) -> None:
        missing = [
            document
            for document in documents
            if not document.summary or document.summary_language != language or document.summary_provider != llm_provider
        ]
        summaries = await asyncio.gather(
            *(
                self.summarization_service.summarize(document, language, llm_provider)
                for document in missing
            )
        )
        for document, summary in zip(missing, summaries, strict=False):
            document.summary = summary
            document.summary_language = language
            document.summary_provider = llm_provider
            if document.cache_key:
                self._document_cache[document.cache_key] = document.model_copy(deep=True)

        for document in documents:
            if document.summary and document.summary_language == language and document.summary_provider == llm_provider:
                continue
            cached = self._document_cache.get(document.cache_key or "")
            if cached and cached.summary and cached.summary_language == language and cached.summary_provider == llm_provider:
                document.summary = cached.summary
                document.summary_language = cached.summary_language
                document.summary_provider = cached.summary_provider

    def _prune_document_cache(self, active_keys: set[str]) -> None:
        removed = [key for key in self._document_cache if key not in active_keys]
        for key in removed:
            del self._document_cache[key]
