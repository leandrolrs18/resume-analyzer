import pytest

from app.services.resume_parser_service import ResumeParserService


@pytest.mark.asyncio
async def test_resume_parser_splits_sections_without_llm() -> None:
    text = """
    Ana Silva
    Formação Acadêmica
    Bacharelado em Ciência da Computação
    Experiência Profissional
    Desenvolvedora Backend na XPTO
    Competências
    Python, FastAPI, Docker
    Projetos
    API de triagem de currículos
    Idiomas
    Português, Inglês
    """

    profile = await ResumeParserService().parse(text)

    assert profile.name == "Ana Silva"
    assert profile.education == ["Bacharelado em Ciência da Computação"]
    assert profile.experience == ["Desenvolvedora Backend na XPTO"]
    assert {"Python", "FastAPI", "Docker"}.issubset(set(profile.skills))
    assert profile.projects == ["API de triagem de currículos"]
