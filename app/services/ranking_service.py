import math
import re
from collections import Counter

from app.schemas import Citation, RankingEvidence, ResumeDocument

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9+#.]+")

QUERY_EXPANSIONS = {
    "estud": {"formação", "academica", "acadêmica", "universidade", "faculdade", "curso"},
    "formacao": {"formação", "graduação", "bacharelado", "especialização", "universidade"},
    "formação": {"graduação", "bacharelado", "especialização", "universidade", "faculdade"},
    "backend": {"api", "python", "django", "fastapi", "docker", "aws", "banco"},
    "python": {"django", "fastapi", "pandas", "numpy", "machine", "learning"},
}


class RankingService:
    def __init__(self, top_k_citations: int = 3):
        self.top_k_citations = top_k_citations

    async def rank(self, query: str, documents: list[ResumeDocument]) -> list[RankingEvidence]:
        chunks = [chunk for document in documents for chunk in document.chunks]
        if not chunks:
            return [
                RankingEvidence(candidate=document.candidate, score=0.0, citations=[])
                for document in documents
            ]

        query_tokens = self._query_tokens(query)
        doc_freq = Counter()
        chunk_tokens = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            chunk_tokens.append((chunk, tokens))
            doc_freq.update(set(tokens))

        total_chunks = len(chunks)
        scored_by_candidate: dict[str, list[tuple[float, Citation]]] = {}
        for chunk, tokens in chunk_tokens:
            score = self._bm25_score(query_tokens, tokens, doc_freq, total_chunks)
            if score <= 0:
                continue
            scored_by_candidate.setdefault(chunk.candidate, []).append(
                (score, Citation(chunk_id=chunk.chunk_id, text=chunk.text))
            )

        evidences = []
        max_score = max(
            (
                sum(score for score, _ in items[: self.top_k_citations])
                for items in scored_by_candidate.values()
            ),
            default=1.0,
        )
        for document in documents:
            candidate_items = sorted(
                scored_by_candidate.get(document.candidate, []),
                key=lambda item: item[0],
                reverse=True,
            )
            raw_score = sum(score for score, _ in candidate_items[: self.top_k_citations])
            citations = [citation for _, citation in candidate_items[: self.top_k_citations]]
            normalized_score = raw_score / max_score if max_score else 0.0
            evidences.append(
                RankingEvidence(
                    candidate=document.candidate,
                    score=round(normalized_score, 4),
                    citations=citations,
                )
            )

        return sorted(evidences, key=lambda item: item.score, reverse=True)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @classmethod
    def _query_tokens(cls, query: str) -> list[str]:
        tokens = cls._tokenize(query)
        expanded = set(tokens)
        for token in tokens:
            for prefix, related in QUERY_EXPANSIONS.items():
                if token.startswith(prefix):
                    expanded.update(related)
        return list(expanded)

    @classmethod
    def _bm25_score(
        cls,
        query_tokens: list[str],
        document_tokens: list[str],
        doc_freq: Counter[str],
        total_documents: int,
    ) -> float:
        if not document_tokens:
            return 0.0
        frequencies = Counter(document_tokens)
        average_length = 120
        k1 = 1.5
        b = 0.75
        score = 0.0
        doc_length = len(document_tokens)
        for token in query_tokens:
            frequency = frequencies[token]
            if frequency == 0:
                continue
            inverse_doc_frequency = math.log(
                1 + (total_documents - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * doc_length / average_length)
            score += inverse_doc_frequency * (frequency * (k1 + 1) / denominator)
        return score
