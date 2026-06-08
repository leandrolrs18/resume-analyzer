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
            # Extração dos currículos:
            # 1) Recebe UploadFile do FastAPI e lê os bytes.
            # 2) DocumentService identifica PDF ou imagem.
            # 3) PDF usa PyMuPDF; imagem ou página escaneada usa Tesseract.
            # 4) Retorna list[ResumeDocument] com extracted_text.
            documents = await self.document_service.extract_documents(files)
            self._log_stage("documents_extracted", stage_started, request_id)

            stage_started = time.perf_counter()
            # Preparação para RAG/ranking:
            # 1) Recebe ResumeDocument com extracted_text.
            # 2) TextChunker usa RecursiveCharacterTextSplitter.
            # 3) Divide o texto em ResumeChunk respeitando tamanho e overlap.
            # 4) Preenche document.chunks para busca por evidências.
            documents = await self._prepare_documents(documents)
            self._log_stage("documents_structured_and_chunked", stage_started, request_id)
            if not query:
                stage_started = time.perf_counter()
                # Fluxo sem query:
                # 1) Cada currículo vai para SummarizationService.
                # 2) O LLM gera um resumo curto baseado no texto extraído.
                # 3) O resumo é salvo em document.summary.
                # 4) A resposta retorna um SummaryResult por candidato.
                await self._summarize_documents(documents, language, llm_provider)
                self._log_stage("documents_summarized", stage_started, request_id)
                results = [
                    SummaryResult(candidate=document.candidate, summary=document.summary or "")
                    for document in documents
                ]
                response = AnalyzeResponse(request_id=request_id, results=results)
            else:
                stage_started = time.perf_counter()
                # Fluxo com query:
                # 1) RankingService compara query contra os chunks.
                # 2) Usa busca híbrida em memória: BM25 + similaridade por embeddings simples.
                # 3) Retorna RankingEvidence com score e citações.
                # 4) Essas evidências alimentam a síntese final.
                evidence = await self.ranking_service.rank(query, documents)
                self._log_stage("in_memory_ranking_completed", stage_started, request_id)

                stage_started = time.perf_counter()
                documents_by_candidate = {document.candidate: document for document in documents}
                # Síntese do ranking:
                # 1) Recebe query, evidências e documentos por candidato.
                # 2) SummarizationService monta prompt com citações relevantes.
                # 3) LLM gera summary e justification sem inventar dados.
                # 4) Resultado vira complemento do RankingResult.
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

            # Serializa a resposta para persistir no log de auditoria.
            response_payload = response.model_dump(mode="json")
            return response
        except Exception:
            status = "failure"
            raise
        finally:
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            try:
                # Auditoria:
                # 1) Salva request_id, user_id, query, resultado e latência.
                # 2) Registra metadados técnicos do pipeline usado.
                # 3) Não persiste os arquivos enviados.
                # 4) Executa mesmo se a análise falhar.
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
        return (
            self.settings.summarization_max_new_tokens + self.settings.justification_max_new_tokens
        )

    async def _prepare_documents(self, documents: list) -> list:
        for document in documents:
            document.chunks = self.chunker.split(document.candidate, document.extracted_text)
        return documents

    async def _summarize_documents(
        self,
        documents: list,
        language: str,
        llm_provider: str,
    ) -> None:
        for document in documents:
            summary = await self.summarization_service.summarize(document, language, llm_provider)
            document.summary = summary
            document.summary_language = language
            document.summary_provider = llm_provider
