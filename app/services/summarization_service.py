import logging
import re
from typing import Any

from app.schemas import RankingEvidence, ResumeDocument

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(
        self, llm_service: Any | None, max_new_tokens: int, groq_service: Any | None = None
    ):
        self.llm_service = llm_service
        self.groq_service = groq_service
        self.max_new_tokens = max_new_tokens

    async def summarize(
        self,
        document: ResumeDocument,
        language: str = "pt",
        llm_provider: str = "local",
    ) -> str:
        fallback = self._fallback_summary(document, language, [])
        summary = fallback
        llm = self._llm(llm_provider)
        if llm:
            try:
                prompt = self._summary_prompt(document, language)
                generated = (await llm.generate(prompt, max(self.max_new_tokens, 220))).strip()
                generated = self._six_line_summary(generated)
                if self._valid_summary(generated) and not self._looks_like_copied_resume(
                    generated, document
                ):
                    summary = generated
            except Exception:
                logger.exception("llm_summary_failed")
        return summary

    async def synthesize_ranked_results(
        self,
        query: str,
        language: str,
        llm_provider: str,
        evidence: list[RankingEvidence],
        documents_by_candidate: dict[str, ResumeDocument],
        max_new_tokens: int,
    ) -> dict[str, dict[str, str]]:
        fallback = {
            item.candidate: {
                "summary": self._fallback_summary(
                    documents_by_candidate[item.candidate],
                    language,
                    [citation.text for citation in item.citations],
                ),
                "justification": self._fallback_justification(query, language, item),
            }
            for item in evidence
        }

        llm = self._llm(llm_provider)
        grounded = [item for item in evidence if item.score > 0 and item.citations]
        if not llm or not grounded:
            return fallback

        try:
            parsed = self._blocks(
                await llm.generate(self._prompt(query, language, grounded), max_new_tokens)
            )
            for candidate, item in parsed.items():
                if candidate not in fallback:
                    continue
                summary = item.get("summary", "").strip()
                if self._valid_summary(summary):
                    fallback[candidate]["summary"] = summary
        except Exception:
            logger.exception("single_llm_synthesis_failed")
        return fallback

    def _llm(self, provider: str):
        if provider == "groq" and self.groq_service:
            return self.groq_service
        return self.llm_service

    @staticmethod
    def _summary_prompt(document: ResumeDocument, language: str) -> str:
        lang = "inglês" if language == "en" else "português do Brasil"
        profile = (
            document.structured_profile.model_dump()
            if document.structured_profile is not None
            else {}
        )
        evidence = document.extracted_text[:3000]
        return (
            f"Responda em {lang}. Use somente os dados do currículo abaixo, sem inventar. "
            "Retorne exatamente 6 linhas. Cada linha deve ser uma frase curta, com no máximo "
            "22 palavras. Cubra formação, experiência, competências, projetos ou atividades "
            "relevantes e senioridade aparente. Evite repetir ideias. Não use markdown, "
            "numeração, bullets, rótulos, dados de contato ou cabeçalho copiado do currículo.\n\n"
            f"Candidato: {document.candidate}\n"
            f"Perfil estruturado: {profile}\n\n"
            f"Trecho do currículo:\n{evidence}"
        )

    @staticmethod
    def _prompt(query: str, language: str, evidence: list[RankingEvidence]) -> str:
        lang = "inglês" if language == "en" else "português do Brasil"
        blocks = []
        for item in evidence[:2]:
            citations = "\n".join(f"- {citation.text[:220]}" for citation in item.citations[:2])
            blocks.append(f"Candidato: {item.candidate}\nEvidências:\n{citations}")
        return (
            f"Responda em {lang}. Use somente as evidências abaixo, sem inventar dados. "
            "Para cada candidato, retorne exatamente este formato, sem JSON e sem markdown:\n"
            "CANDIDATO: nome\n"
            "RESUMO: um parágrafo corrido com 3 frases curtas\n"
            "JUSTIFICATIVA: 1 frase objetiva ligada à pergunta\n"
            "FIM\n\n"
            f"Pergunta: {query}\n\n" + "\n\n".join(blocks)
        )

    @staticmethod
    def _blocks(raw: str) -> dict[str, dict[str, str]]:
        parsed: dict[str, dict[str, str]] = {}
        current: str | None = None
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("CANDIDATO:"):
                current = line.split(":", maxsplit=1)[1].strip()
                parsed[current] = {"summary": "", "justification": ""}
            elif current and line.upper().startswith("RESUMO:"):
                parsed[current]["summary"] = line.split(":", maxsplit=1)[1].strip()
            elif current and line.upper().startswith("JUSTIFICATIVA:"):
                parsed[current]["justification"] = line.split(":", maxsplit=1)[1].strip()
        if not parsed:
            raise ValueError("LLM did not return candidate blocks")
        return parsed

    @classmethod
    def _fallback_summary(
        cls,
        document: ResumeDocument,
        language: str,
        citations: list[str],
    ) -> str:
        profile = document.structured_profile
        skills = profile.skills if profile else []
        if language == "en":
            education_text = cls._sample(profile.education if profile else [], "education")
            experience_text = cls._sample(profile.experience if profile else [], "work")
            project_text = cls._sample(profile.projects if profile else [], "the resume")
            return (
                f"{document.candidate} has academic evidence in "
                f"{education_text} and professional experience in {experience_text}. "
                f"The extracted skills include {cls._join(skills[:6], 'en')}. "
                f"Relevant projects or activities include {project_text}."
            )
        education_text = cls._sample(profile.education if profile else [], "trechos extraídos")
        experience_text = cls._sample(
            profile.experience if profile else [], "atividades profissionais"
        )
        project_text = cls._sample(profile.projects if profile else [], "evidências do currículo")
        return (
            f"{document.candidate} apresenta formação em {education_text} e "
            f"experiência em {experience_text}. "
            f"As competências extraídas incluem {cls._join(skills[:6], 'pt')}. "
            f"Projetos ou atividades relevantes aparecem em {project_text}."
        )

    @classmethod
    def _fallback_justification(cls, query: str, language: str, item: RankingEvidence) -> str:
        if not item.citations:
            if language == "en":
                return f"{item.candidate} did not show strong evidence for the question: {query}."
            return f"{item.candidate} não apresentou evidências fortes para a pergunta: {query}."
        if language == "en":
            citation_text = cls._sample(
                [citation.text for citation in item.citations], "relevant evidence"
            )
            return (
                f"{item.candidate} was ranked for '{query}' because the retrieved citations "
                f"mention {citation_text}."
            )
        citation_text = cls._sample(
            [citation.text for citation in item.citations], "evidências relevantes"
        )
        return (
            f"{item.candidate} foi ranqueado para '{query}' porque as citações recuperadas "
            f"mencionam {citation_text}."
        )

    @staticmethod
    def _level(count: int, language: str) -> str:
        if language == "en":
            return "strong" if count >= 3 else "moderate" if count else "limited"
        return "forte" if count >= 3 else "moderada" if count else "limitada"

    @staticmethod
    def _join(values: list[str], language: str) -> str:
        values = values or (["limited signals"] if language == "en" else ["sinais limitados"])
        if len(values) == 1:
            return values[0]
        return f"{', '.join(values[:-1])}{' and ' if language == 'en' else ' e '}{values[-1]}"

    @staticmethod
    def _sample(values: list[str], fallback: str) -> str:
        clean = [" ".join(value.split())[:120] for value in values if value.strip()]
        return "; ".join(clean[:2]) if clean else fallback

    @staticmethod
    def _valid_summary(value: str) -> bool:
        return len(value) >= 120 and sum(value.count(char) for char in ".!?") >= 2

    @classmethod
    def _looks_like_copied_resume(cls, value: str, document: ResumeDocument) -> bool:
        normalized_value = cls._normalize_for_comparison(value)
        if any(marker in normalized_value for marker in ("@", "linkedin", "github", "portfolio")):
            return True

        copied_lines = 0
        resume_lines = [
            cls._normalize_for_comparison(line)
            for line in document.extracted_text.splitlines()
            if len(cls._normalize_for_comparison(line)) >= 18
        ]
        for generated_line in value.splitlines():
            normalized_line = cls._normalize_for_comparison(generated_line)
            if len(normalized_line) < 18:
                continue
            if any(
                normalized_line == resume_line
                or normalized_line in resume_line
                or resume_line in normalized_line
                for resume_line in resume_lines[:40]
            ):
                copied_lines += 1
            if copied_lines >= 2:
                return True
        return False

    @staticmethod
    def _normalize_for_comparison(value: str) -> str:
        return " ".join(value.casefold().split())

    @staticmethod
    def _six_line_summary(value: str) -> str:
        lines = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            for line in value.splitlines()
            if line.strip()
        ]
        if len(lines) == 1:
            lines = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", lines[0])
                if sentence.strip()
            ]
        return "\n".join(lines[:6])
