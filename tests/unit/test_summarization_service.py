import pytest

from app.schemas import Citation, RankingEvidence, ResumeDocument
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


class RankedLlm:
    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return (
            "CANDIDATO: Ana\n"
            "RESUMO: Ana atua em backend com Python e AWS. Também desenvolve APIs e "
            "pipelines de dados. O perfil apresenta aderência para sistemas distribuídos.\n"
            "JUSTIFICATIVA: Ana se destaca porque as evidências mostram experiência em "
            "backend, APIs e pipelines de dados.\n"
            "FIM"
        )


class RecordingRankedLlm:
    def __init__(self) -> None:
        self.prompt = ""
        self.max_new_tokens = 0

    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        return (
            "CANDIDATO: Ana\n"
            "RESUMO: Ana possui evidências relevantes. O perfil tem aderência parcial. "
            "A análise usa as citações fornecidas.\n"
            "JUSTIFICATIVA: Ana apresenta relação com a consulta a partir das evidências "
            "selecionadas no currículo.\n"
            "FIM\n"
            "CANDIDATO: Bia\n"
            "RESUMO: Bia possui evidências relevantes. O perfil tem aderência parcial. "
            "A análise usa as citações fornecidas.\n"
            "JUSTIFICATIVA: Bia apresenta relação com a consulta a partir das evidências "
            "selecionadas no currículo.\n"
            "FIM\n"
            "CANDIDATO: Caio\n"
            "RESUMO: Caio possui evidências relevantes. O perfil tem aderência parcial. "
            "A análise usa as citações fornecidas.\n"
            "JUSTIFICATIVA: Caio apresenta relação com a consulta a partir das evidências "
            "selecionadas no currículo.\n"
            "FIM"
        )


class FirstPersonJustificationLlm:
    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return (
            "CANDIDATO: Ana\n"
            "RESUMO: Ana atua em backend com Python e AWS. Também desenvolve APIs e "
            "pipelines de dados. O perfil apresenta aderência para sistemas distribuídos.\n"
            "JUSTIFICATIVA: Apliquei experiência em desenvolvimento de sistemas e "
            "aprimorando projetos acadêmicos.\n"
            "FIM"
        )


class EnglishSummaryLlm:
    async def generate(self, prompt: str, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return (
            "Ana has academic evidence in Computer Science.\n"
            "Ana has professional experience with backend development.\n"
            "The extracted skills include Python, AWS and Docker.\n"
            "Relevant projects include APIs and data pipelines.\n"
            "The resume shows strong technical experience."
        )


@pytest.mark.asyncio
async def test_summary_without_query_returns_portuguese_paragraph() -> None:
    service = SummarizationService(llm_service=None, max_new_tokens=120)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Python AWS Docker",
    )

    summary = await service.summarize(document, language="pt")

    sentences = [sentence for sentence in summary.split(".") if sentence.strip()]
    assert "\n" not in summary
    assert 5 <= len(sentences) <= 8
    assert "formação" in summary.casefold()
    assert "Python" in summary
    assert "has academic evidence" not in summary
    assert "perfil baseado nos dados extraídos" not in summary.casefold()


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
    assert "se destacou" in result["Ana"]["justification"]
    assert "was ranked" not in result["Ana"]["justification"]


@pytest.mark.asyncio
async def test_ranked_synthesis_uses_llm_justification_when_valid() -> None:
    service = SummarizationService(llm_service=RankedLlm(), max_new_tokens=120)
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Python AWS backend APIs pipelines de dados",
    )
    result = await service.synthesize_ranked_results(
        query="qual tem mais experiência?",
        language="pt",
        llm_provider="local",
        evidence=[
            RankingEvidence(
                candidate="Ana",
                score=1,
                citations=[Citation(chunk_id="1", text="Experiência em backend e APIs")],
            )
        ],
        documents_by_candidate={"Ana": document},
        max_new_tokens=160,
    )

    assert result["Ana"]["justification"] == (
        "Ana se destaca porque as evidências mostram experiência em backend, APIs e "
        "pipelines de dados."
    )


@pytest.mark.asyncio
async def test_ranked_synthesis_sends_only_candidates_above_half_score_to_llm() -> None:
    llm = RecordingRankedLlm()
    service = SummarizationService(llm_service=llm, max_new_tokens=120)
    documents = {
        name: ResumeDocument(candidate=name, source_filename=f"{name}.pdf", extracted_text=name)
        for name in ("Ana", "Bia", "Caio")
    }

    result = await service.synthesize_ranked_results(
        query="qual candidato pontuou?",
        language="pt",
        llm_provider="local",
        evidence=[
            RankingEvidence(
                candidate="Ana",
                score=1,
                citations=[Citation(chunk_id="1", text="Ana evidencia")],
            ),
            RankingEvidence(
                candidate="Bia",
                score=0.3,
                citations=[Citation(chunk_id="2", text="Bia evidencia")],
            ),
            RankingEvidence(
                candidate="Caio",
                score=0.1,
                citations=[Citation(chunk_id="3", text="Caio evidencia")],
            ),
        ],
        documents_by_candidate=documents,
        max_new_tokens=160,
    )

    assert "Candidato: Ana" in llm.prompt
    assert "Candidato: Bia" not in llm.prompt
    assert "Candidato: Caio" not in llm.prompt
    assert llm.max_new_tokens == 160
    assert result["Ana"]["summary"]
    assert "se destacou" in result["Bia"]["justification"]
    assert "se destacou" in result["Caio"]["justification"]


@pytest.mark.asyncio
async def test_ranked_synthesis_rejects_first_person_justification() -> None:
    service = SummarizationService(
        llm_service=FirstPersonJustificationLlm(),
        max_new_tokens=120,
    )
    document = ResumeDocument(
        candidate="Ana",
        source_filename="ana.pdf",
        extracted_text="Python AWS backend APIs pipelines de dados",
    )
    result = await service.synthesize_ranked_results(
        query="qual tem mais experiência?",
        language="pt",
        llm_provider="local",
        evidence=[
            RankingEvidence(
                candidate="Ana",
                score=1,
                citations=[
                    Citation(
                        chunk_id="1",
                        text=(
                            "Atuei com desenvolvimento de sistemas backend, APIs e pipelines "
                            "de dados em infraestrutura AWS."
                        ),
                    )
                ],
            )
        ],
        documents_by_candidate={"Ana": document},
        max_new_tokens=160,
    )

    assert result["Ana"]["justification"] == (
        "Apliquei experiência em desenvolvimento de sistemas e aprimorando projetos acadêmicos."
    )
