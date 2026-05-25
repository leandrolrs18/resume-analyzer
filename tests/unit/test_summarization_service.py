import pytest

from app.schemas import Citation, RankingEvidence, ResumeDocument, ResumeStructuredProfile
from app.services.summarization_service import SummarizationService


@pytest.mark.asyncio
async def test_summary_is_short_portuguese_paragraph() -> None:
    service = SummarizationService(llm_service=None, max_new_tokens=120)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Python AWS Docker",
        structured_profile=ResumeStructuredProfile(
            education=["Bacharelado em Computação"],
            experience=["Backend Developer"],
            skills=["Python", "AWS", "Docker"],
            projects=["API project"],
        ),
    )

    summary = await service.summarize(document, language="pt")

    assert "\n" not in summary
    assert "formação" in summary
    assert "Python" in summary


@pytest.mark.asyncio
async def test_ranked_fallback_returns_justification() -> None:
    service = SummarizationService(llm_service=None, max_new_tokens=120)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Python AWS",
    )
    result = await service.synthesize_ranked_results(
        query="backend python",
        language="pt",
        llm_provider="local",
        evidence=[
            RankingEvidence(
                candidate="Ana",
                score=1,
                citations=[Citation(chunk_id="1", text="Python AWS backend")],
            )
        ],
        documents_by_candidate={"Ana": document},
        max_new_tokens=120,
    )

    assert "Ana" in result
    assert "foi ranqueado" in result["Ana"]["justification"]
