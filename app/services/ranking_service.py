import re
from collections import Counter

from app.schemas import Citation, RankingEvidence, ResumeDocument

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9+#.]+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d?\s?\d{4}[-\s]?\d{4}")
URL_RE = re.compile(r"https?://\S+")

QUERY_ALIASES = {
    "estud": "formação educação education university college school degree bachelor certificado",
    "form": "formação educação education university college degree bachelor curso",
    "educ": "formação educação education university college degree bachelor curso",
    "exper": "experiência experience trabalho internship intern developer projetos",
    "prepar": "experiência skills projetos software developer",
    "backend": "backend api python django fastapi docker aws",
    "machine": "machine learning python model neural algorithms",
}
SEMANTIC_GROUPS = {
    "formacao": {
        "formação",
        "educação",
        "education",
        "university",
        "college",
        "school",
        "degree",
        "bachelor",
        "certificado",
        "certificate",
    },
    "experiencia": {"experiência", "experience", "intern", "internship", "developer", "work"},
    "skills": {"python", "aws", "docker", "java", "react", "machine", "learning", "sql"},
}


class RankingService:
    def __init__(self, top_k_citations: int = 3):
        self.top_k_citations = top_k_citations

    async def rank(
        self,
        query: str,
        documents: list[ResumeDocument],
        retrieval_mode: str = "hybrid",
    ) -> list[RankingEvidence]:
        query_tokens = self._expand_query(query)
        scored: dict[str, list[tuple[float, Citation]]] = {}

        for document in documents:
            for chunk in document.chunks:
                score = self._score(query_tokens, self._tokens(chunk.text), retrieval_mode)
                if score > 0:
                    scored.setdefault(document.candidate, []).append(
                        (
                            score,
                            Citation(
                                chunk_id=chunk.chunk_id, text=self._clean_citation(chunk.text)
                            ),
                        )
                    )

        return self._results(documents, scored)

    def _results(
        self,
        documents: list[ResumeDocument],
        scored: dict[str, list[tuple[float, Citation]]],
    ) -> list[RankingEvidence]:
        raw_totals = []
        for document in documents:
            items = sorted(
                scored.get(document.candidate, []), reverse=True, key=lambda item: item[0]
            )
            raw_totals.append(sum(score for score, _ in items[: self.top_k_citations]))
        max_total = max(raw_totals, default=0) or 1

        results = []
        for document, total in zip(documents, raw_totals, strict=False):
            items = sorted(
                scored.get(document.candidate, []), reverse=True, key=lambda item: item[0]
            )
            results.append(
                RankingEvidence(
                    candidate=document.candidate,
                    score=round(min(1, total / max_total), 4),
                    citations=[citation for _, citation in items[: self.top_k_citations]],
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _score(self, query_tokens: list[str], chunk_tokens: list[str], mode: str) -> float:
        if not query_tokens or not chunk_tokens:
            return 0
        query = Counter(query_tokens)
        chunk = Counter(chunk_tokens)
        lexical = sum(min(count, chunk[token]) for token, count in query.items())
        semantic = sum(
            1
            for group in SEMANTIC_GROUPS.values()
            if group.intersection(query_tokens) and group.intersection(chunk_tokens)
        )
        if mode == "bm25":
            return lexical
        if mode == "embedding":
            return semantic
        return lexical + (0.7 * semantic)

    @classmethod
    def _expand_query(cls, text: str) -> list[str]:
        tokens = cls._tokens(text)
        expanded = set(tokens)
        for token in tokens:
            for prefix, values in QUERY_ALIASES.items():
                if token.startswith(prefix):
                    expanded.update(cls._tokens(values))
        return list(expanded)

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @staticmethod
    def _clean_citation(text: str) -> str:
        text = URL_RE.sub("[url]", text)
        text = EMAIL_RE.sub("[email]", text)
        text = PHONE_RE.sub("[telefone]", text)
        return " ".join(text.split())[:700].rstrip()
