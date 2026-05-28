import logging
import re
from typing import Any

from app.schemas import RankingEvidence, ResumeDocument

logger = logging.getLogger(__name__)


class SummarizationService:
    def __init__(
        self,
        llm_service: Any | None,
        max_new_tokens: int,
        gemini_service: Any | None = None,
    ):
        self.llm_service = llm_service
        self.gemini_service = gemini_service
        self.max_new_tokens = max_new_tokens

    async def summarize(
        self,
        document: ResumeDocument,
        language: str = "pt",
        llm_provider: str = "local",
    ) -> str:
        language = self._normalize_language(language)
        fallback = self._fallback_summary(document, language, [])
        summary = fallback
        llm = self._llm(llm_provider)
        if llm:
            try:
                prompt = self._summary_prompt(document, language)
                generated = (await llm.generate(prompt, max(self.max_new_tokens, 220))).strip()
                generated = self._summary_lines(generated)
                if self._valid_summary(generated, language) and not self._looks_like_copied_resume(
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
            prompt = self._prompt(query, language, grounded)
            print(
                "[LLM][prompt_context]",
                {
                    "provider": llm_provider,
                    "query": query,
                    "prompt": prompt,
                },
            )
            parsed = self._blocks(await llm.generate(prompt, max_new_tokens))
            for candidate_key, item in parsed.items():
                matched_key = None
                if candidate_key in fallback:
                    matched_key = candidate_key
                else:
                    candidate_key_lower = candidate_key.lower()
                    for f_key in fallback:
                        f_key_lower = f_key.lower()
                        words_f = {w for w in f_key_lower.replace("_", " ").replace("-", " ").split() if len(w) >= 3}
                        words_c = {w for w in candidate_key_lower.replace("_", " ").replace("-", " ").split() if len(w) >= 3}
                        ignored_words = {"cv", "pt", "en", "pdf", "desenvolvedor", "developer", "full", "stack"}
                        words_f = {w for w in words_f if w not in ignored_words}
                        words_c = {w for w in words_c if w not in ignored_words}
                        if words_f & words_c:
                            matched_key = f_key
                            break
                if not matched_key:
                    logger.warning(f"Could not match LLM candidate '{candidate_key}' to any fallback key: {list(fallback.keys())}")
                    continue
                summary = item.get("summary", "").strip()
                if self._valid_summary(summary, language):
                    fallback[matched_key]["summary"] = summary
                justification = item.get("justification", "").strip()
                if self._valid_justification(justification, matched_key):
                    fallback[matched_key]["justification"] = justification
        except Exception:
            logger.exception("single_llm_synthesis_failed")
        return fallback

    def _llm(self, provider: str):
        if provider == "gemini" and self.gemini_service:
            return self.gemini_service
        return self.llm_service

    @staticmethod
    def _summary_prompt(document: ResumeDocument, language: str) -> str:
        lang = "inglês" if language == "en" else "português do Brasil"
        evidence = document.extracted_text[:3000]
        return (
            f"Idioma obrigatório: {lang}. Use somente os dados do currículo abaixo, "
            "sem inventar. "
            "Retorne um único parágrafo natural com 5 a 8 frases curtas. Escreva em terceira "
            "pessoa, como avaliador de recrutamento. Cubra perfil profissional, formação, "
            "experiência, competências, projetos ou senioridade aparente. Não use inglês "
            "quando o idioma obrigatório for português. Não use markdown, numeração, bullets, "
            "rótulos, "
            "dados de contato ou cabeçalho copiado do currículo. Não diga 'perfil baseado nos "
            "dados extraídos'. Não repita a mesma tecnologia várias vezes.\n\n"
            f"Candidato: {document.candidate}\n\n"
            f"Trecho do currículo:\n{evidence}"
        )

    @staticmethod
    def _prompt(query: str, language: str, evidence: list[RankingEvidence]) -> str:
        lang = "inglês" if language == "en" else "português do Brasil"
        blocks = []
        for item in evidence[:2]:
            citations = "\n".join(
                f"- {' '.join(citation.text.split())[:220]}..."
                for citation in item.citations[:2]
            )
            blocks.append(f"Candidato: {item.candidate}\nEvidências:\n{citations}")
        return (
            f"Responda em {lang}. Use somente as evidências abaixo, sem inventar dados. "
            "Escreva como avaliador de recrutamento, sempre em terceira pessoa. "
            "Não copie frases do currículo literalmente. Não use primeira pessoa, como "
            "'eu', 'meu', 'fui', 'atuei', 'apliquei', 'liderei' ou 'tenho'. "
            "Não inclua citações diretas, trechos entre aspas ou cópias literais do currículo na justificativa. "
            "A justificativa deve conter apenas referências indiretas, resumindo a experiência e a aderência conceitualmente. "
            "Na justificativa, comece pelo nome do candidato e explique em detalhes o critério de ranking, "
            "comparando de forma analítica a aderência, senioridade, duração, cargos, projetos "
            "ou complexidade quando houver evidência.\n\n"
            "Para cada candidato, retorne exatamente este formato, sem JSON e sem markdown:\n"
            "CANDIDATO: nome (use exatamente o identificador fornecido após 'Candidato: ')\n"
            "RESUMO: um parágrafo corrido com 3 frases curtas\n"
            "JUSTIFICATIVA: um parágrafo objetivo com 2 a 3 frases explicando detalhadamente o alinhamento do candidato\n"
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
        words = []
        for word in document.extracted_text.split():
            lower = word.lower()
            if "@" in lower or "http" in lower or "linkedin" in lower or "github" in lower or "tel" in lower:
                continue
            words.append(word)
        snippet = " ".join(words[:25]) + "..."
        if language == "en":
            return " ".join([
                f"{document.candidate} presents a professional background from the resume.",
                f"The extracted text snippet includes: {snippet}",
                "No structured profile could be loaded for validation.",
                "Extracted context shows relevant information.",
                "Relevant projects or activities appear in the text.",
                "The profile should be reviewed with the original resume evidence."
            ])
        return " ".join([
            f"{document.candidate} apresenta trajetória profissional descrita no currículo.",
            f"O texto extraído do candidato inclui: {snippet}",
            "A formação e competências constam no arquivo original.",
            "O perfil apresenta aderência técnica para a área.",
            "Projetos e experiências adicionais aparecem no currículo.",
            "A avaliação deve considerar as evidências originais do currículo."
        ])

    @classmethod
    def _fallback_justification(cls, query: str, language: str, item: RankingEvidence) -> str:
        del language
        if not item.citations:
            return f"{item.candidate} não apresentou evidências fortes para a pergunta: {query}."
        return (
            f"{item.candidate} se destacou para \"{query}\" porque há evidências de "
            "experiências e qualificações correlatas em seu histórico profissional."
        )

    @classmethod
    def _valid_summary(cls, value: str, language: str = "pt") -> bool:
        if len(value) < 100:
            return False
        if language == "pt" and any(phrase in value.casefold() for phrase in ("has academic", "professional experience", "extracted skills")):
            return False
        return True

    @classmethod
    def _valid_justification(cls, value: str, candidate: str) -> bool:
        del candidate
        low = value.casefold()
        first_person = (" eu ", " meu ", " atuei ", " fui ", " tenho ", " liderei ", "apliquei")
        return len(low) >= 30 and not any(fp in f" {low} " for fp in first_person)

    @classmethod
    def _looks_like_copied_resume(cls, value: str, document: ResumeDocument) -> bool:
        del document
        normalized_value = value.casefold()
        return any(marker in normalized_value for marker in ("@", "http", "linkedin.com", "github.com"))

    @staticmethod
    def _summary_lines(value: str) -> str:
        lines = []
        for line in value.splitlines():
            clean = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip(" -–:;")
            clean = re.sub(r"^(?:resumo|perfil|formação|experiência|competências):\s*", "", clean, flags=re.I)
            if len(clean) >= 12:
                lines.append(clean)
        return " ".join(lines)

    @staticmethod
    def _normalize_language(language: str) -> str:
        return "en" if language == "en" else "pt"
