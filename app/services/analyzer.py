import asyncio
import logging
import time
from typing import Any

from fastapi import Request, UploadFile

from app.core.config import Settings
from app.core.security import validate_uploads
from app.schemas import AnalyzeResponse, RankingResult, SummaryResult
from app.rag.chunker import TextChunker
from app.repositories.audit_logs import AuditLogRepository
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
        request_id: str,
        user_id: str,
    ) -> AnalyzeResponse:
        validate_uploads(files, self.settings)
        started_at = time.perf_counter()
        status = "success"
        response_payload: dict[str, Any] = {}
        try:
            documents = await self.document_service.extract_documents(files)
            
            # Os resumos iniciais podem continuar assíncronos se o modelo estiver desligado,
            # mas para garantir estabilidade máxima na CPU, o ideal é processar o fluxo do LLM controlado.
            summaries = await asyncio.gather(
                *(self.summarization_service.summarize(document) for document in documents)
            )
            for document, summary in zip(documents, summaries, strict=False):
                document.summary = summary
                document.chunks = self.chunker.split(document.candidate, document.extracted_text)
            
            if not query:
                results = [
                    SummaryResult(candidate=document.candidate, summary=document.summary or "")
                    for document in documents
                ]
                response = AnalyzeResponse(request_id=request_id, results=results)
            else:
                # 1. MUDANÇA INTERNA: O ranking agora roda avaliando semanticamente via LLM e já retorna ordenativo
                evidence = await self.ranking_service.rank(query, documents)
                summaries = {doc.candidate: doc.summary or "" for doc in documents}
                
                # 2. MUDANÇA CRÍTICA: Substituído o asyncio.gather por um loop sequencial para o LLM.
                # Como o Qwen2.5-1.5B consome muita CPU por token gerado, rodar em série evita 
                # que o container congele e estoure a latência alvo de 20s.
                justifications = []
                for item in evidence:
                    justification = await self.summarization_service.justify(
                        query=query,
                        candidate=item.candidate,
                        citations=[citation.text for citation in item.citations],
                        max_new_tokens=self.settings.justification_max_new_tokens,
                    )
                    justifications.append(justification)

                results: list[RankingResult] = []
                for rank, (item, justification) in enumerate(
                    zip(evidence, justifications, strict=False),
                    start=1,
                ):
                    results.append(
                        RankingResult(
                            rank=rank,
                            candidate=item.candidate,
                            score=round(item.score, 4),
                            summary=summaries[item.candidate],
                            justification=justification,
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
                # 3. MUDANÇA DE METADADOS: Atualizado os nomes de auditoria para o relatório do MongoDB 
                # refletir a nova arquitetura correta do projeto pedida na avaliação.
                await self.audit_logs.save_log(
                    request_id=request_id,
                    user_id=user_id,
                    query=query,
                    result=response_payload,
                    latency_ms=latency_ms,
                    status=status,
                    costs={
                        "model": "Qwen2.5-1.5B-Instruct-GGUF",
                        "ranking": "Semantic_LLM_Scoring",
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