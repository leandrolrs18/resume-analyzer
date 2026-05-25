import pytest

from app.schemas import ResumeChunk, ResumeDocument
from app.services.ranking_service import RankingService


@pytest.mark.asyncio
async def test_ranking_prefers_matching_resume_and_redacts_contacts() -> None:
    service = RankingService(top_k_citations=1)
    documents = [
        ResumeDocument(
            candidate="Ana",
            source_filename="ana.pdf",
            extracted_text="Python AWS",
            chunks=[
                ResumeChunk(
                    chunk_id="ana-1",
                    candidate="Ana",
                    text="Python AWS ana@email.com https://site.dev",
                )
            ],
        ),
        ResumeDocument(
            candidate="Bia",
            source_filename="bia.pdf",
            extracted_text="React",
            chunks=[ResumeChunk(chunk_id="bia-1", candidate="Bia", text="React frontend")],
        ),
    ]

    ranked = await service.rank("backend python aws", documents)

    assert ranked[0].candidate == "Ana"
    assert ranked[0].score == 1
    assert "[email]" in ranked[0].citations[0].text
    assert "[url]" in ranked[0].citations[0].text


@pytest.mark.asyncio
async def test_ranking_understands_simple_education_query() -> None:
    service = RankingService(top_k_citations=1)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Education Bachelor degree",
        chunks=[
            ResumeChunk(
                chunk_id="ana-edu",
                candidate="Ana",
                text="Education Bachelor degree in Computer Science",
            )
        ],
    )

    ranked = await service.rank("quem estudou mais?", [document], retrieval_mode="hybrid")

    assert ranked[0].score > 0
