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
    rank: int
    candidate: str
    score: float
    summary: str
    justification: str
    citations: list[Citation]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "rank": 1,
                "candidate": "Maria Silva",
                "score": 0.93,
                "summary": "Engenheira backend com experiência em Python, FastAPI e AWS.",
                "justification": (
                    "Maria Silva combina com a pergunta com base nas evidências extraídas "
                    "sobre Python, AWS e arquitetura de APIs."
                ),
                "citations": [
                    {
                        "chunk_id": "Maria Silva-2",
                        "text": "Desenvolvimento de APIs com Python, FastAPI, Docker e AWS.",
                    }
                ],
            }
        },
    )


class AnalyzeResponse(BaseModel):
    request_id: str
    query: str | None = None
    results: list[RankingResult | SummaryResult]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": "req-001",
                    "query": "backend Python FastAPI Docker",
                    "results": [
                        {
                            "rank": 1,
                            "candidate": "Maria Silva",
                            "score": 0.93,
                            "summary": "Engenheira backend com experiência em Python e AWS.",
                            "justification": (
                                "A candidata apresenta evidências diretas de Python, "
                                "FastAPI e Docker."
                            ),
                            "citations": [
                                {
                                    "chunk_id": "Maria Silva-2",
                                    "text": "APIs com Python, FastAPI, Docker e AWS.",
                                }
                            ],
                        }
                    ],
                },
                {
                    "request_id": "req-002",
                    "query": None,
                    "results": [
                        {
                            "candidate": "Joao Souza",
                            "summary": (
                                "Desenvolvedor full stack com experiência em React e Node.js."
                            ),
                        }
                    ],
                },
            ]
        }
    )


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


class ResumeStructuredProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    education: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)


class ResumeDocument(BaseModel):
    candidate: str
    source_filename: str
    extracted_text: str
    cache_key: str | None = None
    summary: str | None = None
    summary_language: str | None = None
    summary_provider: str | None = None
    structured_profile: ResumeStructuredProfile | None = None
    chunks: list[ResumeChunk] = Field(default_factory=list)


class RankingEvidence(BaseModel):
    candidate: str
    score: float
    citations: list[Citation]


class AnalyzeRequestContext(BaseModel):
    request_id: str
    user_id: str
    query: str | None = None
