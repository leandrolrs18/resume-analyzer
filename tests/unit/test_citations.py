import pytest

from app.models.schemas import ResumeChunk, ResumeDocument
from app.services.ranking_service import RankingService


@pytest.mark.asyncio
async def test_ranking_returns_top_citations() -> None:
    service = RankingService(top_k_citations=1)
    document = ResumeDocument(
        candidate="Maria",
        source_filename="maria.pdf",
        extracted_text="text",
        chunks=[
            ResumeChunk(chunk_id="Maria-0", candidate="Maria", text="Python and AWS"),
            ResumeChunk(chunk_id="Maria-1", candidate="Maria", text="FastAPI and Docker"),
        ],
    )

    ranked = await service.rank("Python AWS", [document])

    assert len(ranked[0].citations) == 1
    assert ranked[0].citations[0].chunk_id == "Maria-0"
