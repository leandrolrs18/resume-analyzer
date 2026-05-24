import pytest

from app.schemas import ResumeDocument
from app.services.summarization_service import SummarizationService


@pytest.mark.asyncio
async def test_extractive_summary_skips_contact_header() -> None:
    service = SummarizationService(llm_service=None, max_new_tokens=120)
    document = ResumeDocument(
        candidate="Leandro",
        source_filename="cv.pdf",
        extracted_text="""
        Leandro Rodrigues
        Engenheiro de Inteligência Artificial | Desenvolvedor Backend
        Parnamirim - RN, Brasil
        (84) 9 9841-9659
        leandro@example.com
        Atuação sólida em IA aplicada, NLP e sistemas backend escaláveis com Python e Django.
        Cloud & DevOps AWS, Docker e AirFlow.
        Experiência profissional como Desenvolvedor de IA em projeto industrial.
        Desenvolvimento backend em Python e Django, com uso de IA em Visão Computacional.
        Formação em Tecnologia da Informação.
        Idiomas Português nativo e Inglês avançado.
        """,
    )

    summary = await service.summarize(document)

    assert "leandro@example.com" not in summary
    assert "(84)" not in summary
    assert "Atuação sólida" in summary
    assert "AWS, Docker" in summary
    assert len(summary.splitlines()) <= 6


def test_extractive_justification_is_in_portuguese() -> None:
    justification = SummarizationService._extractive_justification(
        query="backend python",
        language="pt",
        candidate="Maria",
        citations=["Desenvolvimento backend em Python e FastAPI."],
    )

    assert "combina com a pergunta" in justification
    assert "evidências extraídas" in justification
