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
                f"- {SummarizationService._truncate_words(citation.text, 220)}"
                for citation in item.citations[:2]
            )
            blocks.append(f"Candidato: {item.candidate}\nEvidências:\n{citations}")
        return (
            f"Responda em {lang}. Use somente as evidências abaixo, sem inventar dados. "
            "Escreva como avaliador de recrutamento, sempre em terceira pessoa. "
            "Não copie frases do currículo literalmente. Não use primeira pessoa, como "
            "'eu', 'meu', 'fui', 'atuei', 'apliquei', 'liderei' ou 'tenho'. "
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
        citation_text = cls._evidence_summary([citation.text for citation in item.citations])
        return (
            f"{item.candidate} se destacou para \"{query}\" porque há evidências de "
            f"{citation_text}."
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

    @classmethod
    def _valid_summary(cls, value: str, language: str = "pt") -> bool:
        sentences = cls._sentences(value)
        if not 5 <= len(sentences) <= 8:
            return False
        if len(value) < 120 or sum(value.count(char) for char in ".!?") < 4:
            return False
        if "perfil baseado nos dados extraídos" in value.casefold():
            return False
        return language == "en" or not cls._looks_like_english_summary(value)

    @classmethod
    def _valid_justification(cls, value: str, candidate: str) -> bool:
        normalized = " ".join(value.split()).casefold()
        if len(normalized) < 40:
            return False
        blocked = ("was ranked", "retrieved citations", "because the retrieved")
        if any(marker in normalized for marker in blocked):
            return False
        contact_markers = ("@", "linkedin", "github", "telefone", "[email]", "[url]")
        if any(marker in normalized for marker in contact_markers):
            return False
        first_person = (
            " eu ",
            " meu ",
            " minha ",
            " meus ",
            " minhas ",
            "apliquei",
            "fui ",
            "atuei ",
            "contribuí",
            "liderei",
            "tenho ",
            "implementei",
            "desenvolvi",
        )
        padded = f" {normalized} "
        if any(marker in padded for marker in first_person):
            return False
        candidate_name = cls._candidate_reference(candidate)
        return (
            candidate_name in normalized
            or "candidato" in normalized
            or "candidata" in normalized
        )

    @classmethod
    def _evidence_summary(cls, values: list[str]) -> str:
        categories = cls._evidence_categories(values)
        if categories:
            return cls._join(categories[:3], "pt")
        snippets = cls._evidence_snippets(values)
        if snippets:
            return cls._join(snippets[:2], "pt")
        return "evidências relevantes no currículo"

    @classmethod
    def _evidence_categories(cls, values: list[str]) -> list[str]:
        text = " ".join(cls._clean_evidence_text(value).casefold() for value in values)
        categories = []
        checks = (
            (
                "experiência em desenvolvimento de sistemas",
                ("desenvolvimento de sistemas", "software engineer", "developer"),
            ),
            ("atuação em backend e APIs", ("backend", "back-end", "api", "apis")),
            (
                "projetos de dados, pipelines ou infraestrutura",
                ("pipeline", "airflow", "kafka", "infraestrutura", "dados"),
            ),
            (
                "participação em projetos acadêmicos ou plataformas web",
                ("acadêmic", "plataforma", "projeto", "sistemas acadêmicos"),
            ),
            ("uso de tecnologias relevantes para a vaga", ("python", "aws", "docker", "sql")),
            ("liderança ou responsabilidade técnica", ("liderei", "líder", "responsável")),
        )
        for label, markers in checks:
            if any(marker in text for marker in markers):
                categories.append(label)
        return categories

    @classmethod
    def _evidence_snippets(cls, values: list[str]) -> list[str]:
        candidates: list[tuple[int, str]] = []
        for value in values:
            cleaned = cls._clean_evidence_text(value)
            for part in re.split(r"(?<=[.!?])\s+|;\s+|•\s+", cleaned):
                snippet = " ".join(part.split()).strip(" -–:;,.")
                if len(snippet) < 24 or cls._looks_like_contact(snippet):
                    continue
                if snippet[:1].islower():
                    continue
                score = cls._evidence_score(snippet)
                candidates.append((score, cls._truncate_words(snippet, 180)))
        candidates.sort(reverse=True, key=lambda item: (item[0], len(item[1])))
        snippets: list[str] = []
        for _, snippet in candidates:
            if snippet not in snippets:
                snippets.append(snippet)
            if len(snippets) >= 2:
                break
        return snippets

    @staticmethod
    def _clean_evidence_text(value: str) -> str:
        value = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", " ", value)
        value = re.sub(r"https?://\S+", " ", value)
        value = re.sub(r"\b(?:linkedin|github)\b\s*:?\s*\S*", " ", value, flags=re.I)
        value = value.replace("[email]", " ").replace("[url]", " ")
        return " ".join(value.split())

    @staticmethod
    def _looks_like_contact(value: str) -> bool:
        lowered = value.casefold()
        return any(marker in lowered for marker in ("@", "linkedin", "github", "telefone"))

    @staticmethod
    def _evidence_score(value: str) -> int:
        lowered = value.casefold()
        markers = (
            "experiência",
            "experience",
            "desenvolv",
            "backend",
            "back-end",
            "full-stack",
            "lider",
            "pipeline",
            "sistema",
            "projeto",
            "escala",
            "dados",
        )
        return sum(1 for marker in markers if marker in lowered)

    @staticmethod
    def _truncate_words(value: str, limit: int) -> str:
        normalized = " ".join(value.split())
        if len(normalized) <= limit:
            return normalized
        truncated = normalized[:limit].rsplit(" ", maxsplit=1)[0].rstrip(" ,;:.")
        return f"{truncated}..."

    @staticmethod
    def _candidate_reference(candidate: str) -> str:
        words = [word.casefold() for word in candidate.split() if len(word) >= 3]
        return words[0] if words else candidate.casefold()

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
    def _summary_lines(value: str) -> str:
        lines = [
            re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip(" -–:;")
            for line in value.splitlines()
            if line.strip()
        ]
        if len(lines) == 1:
            lines = [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", lines[0])
                if sentence.strip()
            ]
        cleaned: list[str] = []
        seen = set()
        for line in lines:
            line = re.sub(
                r"^(?:resumo|perfil|formação|experiência|competências):\s*",
                "",
                line,
                flags=re.I,
            )
            line = " ".join(line.split())
            normalized = line.casefold()
            if len(line) < 12 or normalized in seen:
                continue
            cleaned.append(line)
            seen.add(normalized)
            if len(cleaned) >= 8:
                break
        return " ".join(cleaned)

    @staticmethod
    def _normalize_language(language: str) -> str:
        return "en" if language == "en" else "pt"

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        unique_values = []
        seen = set()
        for value in values:
            normalized = " ".join(value.split()).casefold()
            if not normalized or normalized in seen:
                continue
            unique_values.append(value)
            seen.add(normalized)
        return unique_values

    @staticmethod
    def _looks_like_english_summary(value: str) -> bool:
        normalized = f" {' '.join(value.casefold().split())} "
        markers = (
            " has academic evidence ",
            " professional experience ",
            " extracted skills ",
            " relevant projects ",
            " the resume ",
            " work.",
            " and ",
            " with ",
        )
        return sum(1 for marker in markers if marker in normalized) >= 2

    @staticmethod
    def _sentences(value: str) -> list[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", " ".join(value.split()))
            if sentence.strip()
        ]
