import json
import logging
import re
from typing import Any

from app.schemas import RankingEvidence, ResumeDocument

logger = logging.getLogger(__name__)

MAX_SUMMARY_SOURCE_CHARS = 3500
SUMMARY_LINES = 6
CONTACT_RE = re.compile(r"(@|\(?\d{2}\)?\s?\d?\s?\d{4}[-\s]?\d{4}|linkedin|github)", re.I)
ROLE_TERMS = {
    "analista",
    "backend",
    "cientista",
    "dados",
    "desenvolvedor",
    "developer",
    "engineer",
    "engenheiro",
    "full stack",
    "inteligência artificial",
    "software",
}
SUMMARY_TERMS = {
    "api",
    "aws",
    "backend",
    "cloud",
    "docker",
    "experiência",
    "formação",
    "ia",
    "machine learning",
    "python",
    "tecnologias",
}


class SummarizationService:
    def __init__(self, llm_service: Any | None, max_new_tokens: int):
        self.llm_service = llm_service
        self.max_new_tokens = max_new_tokens

    async def summarize(self, document: ResumeDocument) -> str:
        return self._extractive_summary(document.extracted_text[:MAX_SUMMARY_SOURCE_CHARS])

    async def synthesize_ranked_results(
        self,
        query: str,
        evidence: list[RankingEvidence],
        documents_by_candidate: dict[str, ResumeDocument],
        max_new_tokens: int,
    ) -> dict[str, dict[str, str]]:
        fallback = {
            item.candidate: {
                "summary": documents_by_candidate[item.candidate].summary
                or self._extractive_summary(documents_by_candidate[item.candidate].extracted_text),
                "justification": self._extractive_justification(
                    query,
                    item.candidate,
                    [citation.text for citation in item.citations],
                ),
            }
            for item in evidence
        }
        if self.llm_service is None or not evidence:
            return fallback

        prompt = self._ranked_prompt(query, evidence, documents_by_candidate)
        try:
            raw = await self.llm_service.generate(prompt, max_new_tokens)
            parsed = self._parse_json_object(raw)
            candidates = parsed.get("candidates", [])
            for item in candidates:
                candidate = str(item.get("candidate", "")).strip()
                if candidate not in fallback:
                    continue
                summary = str(item.get("summary", "")).strip()
                justification = str(item.get("justification", "")).strip()
                if summary:
                    fallback[candidate]["summary"] = summary
                if justification:
                    fallback[candidate]["justification"] = justification
            return fallback
        except Exception:
            logger.exception("single_llm_synthesis_failed")
            return fallback

    @staticmethod
    def _ranked_prompt(
        query: str,
        evidence: list[RankingEvidence],
        documents_by_candidate: dict[str, ResumeDocument],
    ) -> str:
        candidates = []
        for rank, item in enumerate(evidence[:3], start=1):
            document = documents_by_candidate[item.candidate]
            citations = "\n".join(f"- {citation.text[:450]}" for citation in item.citations[:2])
            fallback_summary = document.summary or ""
            candidates.append(
                f"Rank {rank}\n"
                f"Candidato: {item.candidate}\n"
                f"Score: {item.score}\n"
                f"Resumo-base: {fallback_summary[:500]}\n"
                f"Evidências:\n{citations}"
            )

        return (
            "Você é um recrutador técnico. Use somente as evidências fornecidas.\n"
            "Responda em português do Brasil, sem inventar dados.\n"
            "Para cada candidato, escreva sumário e justificativa com uma frase cada.\n"
            "Retorne apenas JSON válido neste formato:\n"
            '{"candidates":[{"candidate":"nome exato","summary":"resumo curto",'
            '"justification":"justificativa objetiva"}]}\n\n'
            f"Pergunta: {query}\n\n"
            "Candidatos e evidências:\n"
            f"{chr(10).join(candidates)}"
        )

    @staticmethod
    def _parse_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM did not return a JSON object")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _extractive_summary(text: str) -> str:
        candidates = []
        for index, line in enumerate(
            SummarizationService._clean_line(line) for line in text.splitlines()
        ):
            if not line or CONTACT_RE.search(line):
                continue
            score = SummarizationService._summary_score(line)
            if score > 0:
                candidates.append((score, index, line))
        if not candidates:
            clean = [SummarizationService._clean_line(line) for line in text.splitlines()]
            clean = [line for line in clean if line and not CONTACT_RE.search(line)]
            return "\n".join(clean[:SUMMARY_LINES]) or "Texto insuficiente para gerar sumário."
        selected = [
            line
            for _, _, line in sorted(candidates, key=lambda item: (-item[0], item[1]))[
                :SUMMARY_LINES
            ]
        ]
        order = {line: index for _, index, line in candidates}
        return "\n".join(sorted(dict.fromkeys(selected), key=lambda line: order[line]))

    @staticmethod
    def _extractive_justification(query: str, candidate: str, citations: list[str]) -> str:
        if not citations:
            return (
                f"{candidate} não apresentou evidências fortes para responder "
                f"à pergunta: {query}."
            )
        evidence = " ".join(citation.strip() for citation in citations if citation.strip())
        evidence = evidence[:900].rstrip()
        return (
            f"{candidate} combina com a pergunta '{query}' "
            f"com base nas evidências extraídas: {evidence}."
        )

    @staticmethod
    def _clean_line(line: str) -> str:
        return " ".join(line.replace("●", "").replace("\u200b", " ").split())

    @staticmethod
    def _summary_score(line: str) -> int:
        normalized = line.lower()
        score = 0
        score += sum(3 for term in ROLE_TERMS if term in normalized)
        score += sum(2 for term in SUMMARY_TERMS if term in normalized)
        if re.search(r"\b(20\d{2}|19\d{2})\b", line):
            score += 1
        if 35 <= len(line) <= 280:
            score += 1
        return score
