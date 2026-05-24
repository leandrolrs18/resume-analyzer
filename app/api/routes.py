from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.schemas import AnalyzeResponse, AuditLogResponse, HealthResponse
from app.services.dependencies import Container, get_container

router = APIRouter()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analisar currículos",
    description=(
        "Recebe currículos PDF/JPEG/PNG, request_id, user_id e uma query opcional. "
        "O processamento é stateless: arquivos e vetores não são persistidos. "
        "Sem query, retorna sumários; com query, retorna ranking, score, justificativas e citações."
    ),
    response_description="Resultado da análise dos currículos",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "ranking": {
                            "summary": "Ranking com query de recrutamento",
                            "value": {
                                "request_id": "req-001",
                                "user_id": "recrutador-demo",
                                "query": "backend Python FastAPI Docker AWS",
                                "files": ["maria.pdf", "joao.png"],
                            },
                        },
                        "summary": {
                            "summary": "Sumário sem query",
                            "value": {
                                "request_id": "req-002",
                                "user_id": "recrutador-demo",
                                "files": ["maria.pdf"],
                            },
                        },
                    }
                }
            }
        }
    },
)
async def analyze_resumes(
    request: Request,
    files: list[UploadFile] = File(...),
    query: str | None = Form(default=None),
    language: str | None = Form(default="pt"),
    request_id: str = Form(...),
    user_id: str = Form(...),
    container: Container = Depends(get_container),
) -> AnalyzeResponse:
    return await container.analyzer.analyze(
        request=request,
        files=files,
        query=query,
        language=language or "pt",
        request_id=request_id,
        user_id=user_id,
    )


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Verificar saúde da API",
    description="Retorna o estado da API e valida a conexão de auditoria quando configurada.",
)
async def healthz(container: Container = Depends(get_container)) -> HealthResponse:
    return await container.healthcheck.run()


@router.get(
    "/logs/{request_id}",
    response_model=AuditLogResponse,
    summary="Consultar logs de auditoria por requisição",
    description="Retorna registros da coleção audit_logs para o request_id informado.",
)
async def get_logs(
    request_id: str,
    container: Container = Depends(get_container),
) -> AuditLogResponse:
    return await container.audit_logs.get_logs(request_id)
