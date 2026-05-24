import pytest

from app.schemas import ResumeChunk, ResumeDocument
from app.services.ranking_service import RankingService


@pytest.mark.asyncio
async def test_in_memory_ranking_returns_top_match() -> None:
    service = RankingService(top_k_citations=1)
    documents = [
        ResumeDocument(
            candidate="Maria",
            source_filename="maria.pdf",
            extracted_text="python",
            chunks=[ResumeChunk(chunk_id="a", candidate="Maria", text="python aws")],
        ),
        ResumeDocument(
            candidate="Joao",
            source_filename="joao.pdf",
            extracted_text="java",
            chunks=[ResumeChunk(chunk_id="b", candidate="Joao", text="java spring")],
        ),
    ]

    results = await service.rank("python", documents)

    assert results[0].candidate == "Maria"
    assert results[0].citations[0].chunk_id == "a"
