import pytest
import math

from app.schemas import Citation, ResumeChunk, ResumeDocument
from app.services.ranking_service import RankingService


# Garante que o ranking favorece o currículo mais aderente e redige contatos nas citações.
@pytest.mark.asyncio
async def test_ranking_prefers_matching_resume_and_redacts_contacts(monkeypatch) -> None:
    monkeypatch.setenv("USE_CROSS_ENCODER_RERANK", "false")
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
    assert 0 < ranked[0].score <= 1
    assert ranked[0].score > ranked[1].score
    assert "[email]" in ranked[0].citations[0].text
    assert "[url]" in ranked[0].citations[0].text


# Garante que a busca lexical-semântica responde a uma query simples de formação.
@pytest.mark.asyncio
async def test_ranking_understands_simple_education_query(monkeypatch) -> None:
    monkeypatch.setenv("USE_CROSS_ENCODER_RERANK", "false")
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

    ranked = await service.rank("education degree", [document])

    assert ranked[0].score > 0


# Garante que um NaN vindo do CrossEncoder não contamina o score nem vira 100%.
def test_rerank_nan_score_becomes_zero(monkeypatch) -> None:
    monkeypatch.setenv("USE_CROSS_ENCODER_RERANK", "true")
    service = RankingService(top_k_citations=1)

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [math.nan, 2.0]

    monkeypatch.setattr(service, "_get_rerank_model", lambda: FakeCrossEncoder())

    scores = service._rerank_scores(
        "professor",
        ["texto com OCR ruim", "texto valido"],
        [0.4, 0.5],
    )

    assert scores[0] == 0.0
    assert 0 < scores[1] <= 1


# Garante que uma falha total do CrossEncoder volta ao score híbrido inicial.
def test_rerank_all_nan_scores_use_fallback_scores(monkeypatch) -> None:
    monkeypatch.setenv("USE_CROSS_ENCODER_RERANK", "true")
    service = RankingService(top_k_citations=1)

    class FakeCrossEncoder:
        def predict(self, pairs):
            return [math.nan, math.nan]

    monkeypatch.setattr(service, "_get_rerank_model", lambda: FakeCrossEncoder())

    scores = service._rerank_scores(
        "professor",
        ["texto um", "texto dois"],
        [0.56, 0.43],
    )

    assert scores == [0.56, 0.43]


# Garante que o cálculo final do candidato também neutraliza NaN por segurança.
def test_candidate_final_score_never_turns_nan_into_one() -> None:
    service = RankingService(top_k_citations=1)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="texto",
        chunks=[],
    )

    results = service._results(
        [document],
        {"Ana": [(math.nan, Citation(chunk_id="ana-1", text="texto"))]},
    )

    assert results[0].score == 0.0
