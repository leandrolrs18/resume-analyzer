import pytest

from app.schemas import Citation, RankingEvidence, ResumeDocument, ResumeStructuredProfile
from app.services.summarization_service import SummarizationService


class EchoLlm:
    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return (
            "FÁBIO VICENTE DE SENA\n"
            "Desenvolvedor Front-end | React.js • TypeScript • Node.js • Firebase\n"
            "Teodoro Sampaio – SP • (18) 98157-9318 • fabiosena1436@gmail.com\n"
            "LinkedIn: linkedin/fabio-vicente-de-sena • GitHub: Github.com/fabiosena1436\n"
            "Portfólio: Portifoliofabiosena.com.br\n"
            "RESUMO PROFISSIONAL."
        )


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
async def test_summary_rejects_copied_resume_header() -> None:
    service = SummarizationService(llm_service=EchoLlm(), max_new_tokens=120)
    document = ResumeDocument(
        candidate="Fábio Vicente De Sena",
        source_filename="fabio.pdf",
        extracted_text=(
            "FÁBIO VICENTE DE SENA\n"
            "Desenvolvedor Front-end | React.js • TypeScript • Node.js • Firebase\n"
            "Teodoro Sampaio – SP • (18) 98157-9318 • fabiosena1436@gmail.com\n"
            "LinkedIn: linkedin/fabio-vicente-de-sena • GitHub: Github.com/fabiosena1436\n"
            "Portfólio: Portifoliofabiosena.com.br\n"
            "RESUMO PROFISSIONAL\n"
            "Atua no desenvolvimento de interfaces web com React e TypeScript."
        ),
        structured_profile=ResumeStructuredProfile(
            experience=["Desenvolvedor Front-end"],
            skills=["React.js", "TypeScript", "Node.js", "Firebase"],
        ),
    )

    summary = await service.summarize(document, language="pt")

    assert "fabiosena1436@gmail.com" not in summary
    assert "GitHub" not in summary
    assert "competências" in summary


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
