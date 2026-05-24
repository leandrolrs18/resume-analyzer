from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    text: str


class SummaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: str
    summary: str


class RankingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int
    candidate: str
    score: float
    summary: str
    justification: str
    citations: list[Citation]


class AnalyzeResponse(BaseModel):
    request_id: str
    query: str | None = None
    results: list[RankingResult | SummaryResult]


class HealthResponse(BaseModel):
    status: str = "ok"


class AuditLogEntry(BaseModel):
    request_id: str
    user_id: str
    timestamp: datetime
    query: str | None = None
    result: dict[str, Any]
    latency_ms: float
    costs: dict[str, Any] = Field(default_factory=dict)
    status: str


class AuditLogResponse(BaseModel):
    request_id: str
    logs: list[AuditLogEntry]


class ResumeChunk(BaseModel):
    chunk_id: str
    candidate: str
    text: str


class ResumeDocument(BaseModel):
    candidate: str
    source_filename: str
    extracted_text: str
    summary: str | None = None
    chunks: list[ResumeChunk] = Field(default_factory=list)


class RankingEvidence(BaseModel):
    candidate: str
    score: float
    citations: list[Citation]


class AnalyzeRequestContext(BaseModel):
    request_id: str
    user_id: str
    query: str | None = None
