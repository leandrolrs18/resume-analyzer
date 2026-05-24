import pytest

from app.schemas import ResumeChunk, ResumeDocument
from app.services.ranking_service import RankingService


@pytest.mark.asyncio
async def test_ranking_prefers_most_similar_candidate() -> None:
    ranking = RankingService(top_k_citations=2)
    documents = [
        ResumeDocument(
            candidate="Maria",
            source_filename="maria.pdf",
            extracted_text="Strong Python and AWS background",
            chunks=[
                ResumeChunk(
                    chunk_id="Maria-0", candidate="Maria", text="Strong Python and AWS background"
                )
            ],
        ),
        ResumeDocument(
            candidate="Joao",
            source_filename="joao.pdf",
            extracted_text="React frontend experience",
            chunks=[
                ResumeChunk(chunk_id="Joao-0", candidate="Joao", text="React frontend experience")
            ],
        ),
    ]

    ranked = await ranking.rank("Python AWS", documents)

    assert ranked[0].candidate == "Maria"
    assert ranked[0].score > ranked[1].score
    assert ranked[0].citations[0].chunk_id == "Maria-0"


@pytest.mark.asyncio
async def test_ranking_expands_education_question() -> None:
    ranking = RankingService(top_k_citations=1)
    documents = [
        ResumeDocument(
            candidate="Maria",
            source_filename="maria.pdf",
            extracted_text="Formação em Ciência da Computação pela Universidade Federal.",
            chunks=[
                ResumeChunk(
                    chunk_id="Maria-0",
                    candidate="Maria",
                    text="Formação em Ciência da Computação pela Universidade Federal.",
                )
            ],
        ),
        ResumeDocument(
            candidate="Joao",
            source_filename="joao.pdf",
            extracted_text="Experiência com React e design system.",
            chunks=[
                ResumeChunk(
                    chunk_id="Joao-0",
                    candidate="Joao",
                    text="Experiência com React e design system.",
                )
            ],
        ),
    ]

    ranked = await ranking.rank("onde estudou?", documents)

    assert ranked[0].candidate == "Maria"
    assert ranked[0].score > 0
    assert "Universidade" in ranked[0].citations[0].text


@pytest.mark.asyncio
async def test_ranking_never_returns_score_above_one() -> None:
    ranking = RankingService(top_k_citations=2)
    document = ResumeDocument(
        candidate="Maria",
        source_filename="maria.pdf",
        extracted_text="Python AWS Docker FastAPI",
        chunks=[
            ResumeChunk(chunk_id="Maria-0", candidate="Maria", text="Python"),
            ResumeChunk(chunk_id="Maria-1", candidate="Maria", text="Python AWS Docker FastAPI"),
            ResumeChunk(chunk_id="Maria-2", candidate="Maria", text="AWS Docker FastAPI"),
        ],
    )

    ranked = await ranking.rank("Python AWS Docker FastAPI", [document])

    assert ranked[0].score <= 1.0


@pytest.mark.asyncio
async def test_ranking_redacts_contact_data_from_citations() -> None:
    ranking = RankingService(top_k_citations=1)
    document = ResumeDocument(
        candidate="Maria",
        source_filename="maria.pdf",
        extracted_text="text",
        chunks=[
            ResumeChunk(
                chunk_id="Maria-0",
                candidate="Maria",
                text=(
                    "Maria maria@email.com (84) 99999-9999 https://portfolio.dev. "
                    "Arquitetura de software com Python, AWS, Docker e FastAPI."
                ),
            )
        ],
    )

    ranked = await ranking.rank("arquiteto software Python AWS", [document])

    citation = ranked[0].citations[0].text
    assert "maria@email.com" not in citation
    assert "99999-9999" not in citation
    assert "https://portfolio.dev" not in citation
    assert "Arquitetura de software" in citation


@pytest.mark.asyncio
async def test_ranking_expands_generic_portuguese_experience_query() -> None:
    ranking = RankingService(top_k_citations=1)
    documents = [
        ResumeDocument(
            candidate="Ana",
            source_filename="ana.pdf",
            extracted_text="Machine learning projects with Python and software development.",
            chunks=[
                ResumeChunk(
                    chunk_id="Ana-0",
                    candidate="Ana",
                    text="Machine learning projects with Python and software development.",
                )
            ],
        ),
        ResumeDocument(
            candidate="Bia",
            source_filename="bia.pdf",
            extracted_text="Customer service and retail operations.",
            chunks=[
                ResumeChunk(
                    chunk_id="Bia-0",
                    candidate="Bia",
                    text="Customer service and retail operations.",
                )
            ],
        ),
    ]

    ranked = await ranking.rank("qual mais preparado com boas experiências?", documents)

    assert ranked[0].candidate == "Ana"
    assert ranked[0].score > 0
