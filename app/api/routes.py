from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.models.schemas import AnalyzeResponse, AuditLogResponse, HealthResponse
from app.services.dependencies import Container, get_container

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_resumes(
    request: Request,
    files: list[UploadFile] = File(...),
    query: str | None = Form(default=None),
    request_id: str = Form(...),
    user_id: str = Form(...),
    container: Container = Depends(get_container),
) -> AnalyzeResponse:
    return await container.analyzer.analyze(
        request=request,
        files=files,
        query=query,
        request_id=request_id,
        user_id=user_id,
    )


@router.get("/healthz", response_model=HealthResponse)
async def healthz(container: Container = Depends(get_container)) -> HealthResponse:
    return await container.healthcheck.run()


@router.get("/logs/{request_id}", response_model=AuditLogResponse)
async def get_logs(
    request_id: str,
    container: Container = Depends(get_container),
) -> AuditLogResponse:
    return await container.audit_logs.get_logs(request_id)
