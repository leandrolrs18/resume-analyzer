import math
import re
from collections import Counter

from app.schemas import Citation, RankingEvidence, ResumeDocument

BM25_WEIGHT = 0.3
EMBEDDING_WEIGHT = 0.7

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
    ) -> list[RankingEvidence]:
        query_tokens = self._expand_query(query)
        chunk_items = [
            (document, chunk, self._tokens(chunk.text))
            for document in documents
            for chunk in document.chunks
        ]
        bm25_scores = self._bm25_scores(query_tokens, [tokens for _, _, tokens in chunk_items])
        max_bm25 = max(bm25_scores, default=0) or 1
        query_embedding = self._embedding_vector(query_tokens)
        scored: dict[str, list[tuple[float, Citation]]] = {}

        initial_candidates = []
        for (document, chunk, chunk_tokens), bm25_raw in zip(
            chunk_items, bm25_scores, strict=False
        ):
            bm25_score = bm25_raw / max_bm25
            embedding_score = self._cosine(
                query_embedding,
                self._embedding_vector(chunk_tokens),
            )
            final_score = (EMBEDDING_WEIGHT * embedding_score) + (BM25_WEIGHT * bm25_score)
            item = {
                "candidate": document.candidate,
                "chunk": chunk,
                "chunk_tokens": chunk_tokens,
                "bm25_raw": bm25_raw,
                "bm25_score": bm25_score,
                "embedding_score": embedding_score,
                "final_score": final_score,
            }
            initial_candidates.append(item)

        top_k_initial = sorted(
            (item for item in initial_candidates if item["final_score"] > 0),
            reverse=True,
            key=lambda item: item["final_score"],
        )[: self._initial_top_k(len(documents))]

        reranked = sorted(
            (
                {
                    **item,
                    "rerank_score": self._rerank_score(
                        item["final_score"],
                        query_tokens,
                        item["chunk_tokens"],
                    ),
                }
                for item in top_k_initial
            ),
            reverse=True,
            key=lambda item: item["rerank_score"],
        )

        for item in reranked:
            scored.setdefault(item["candidate"], []).append(
                (
                    item["rerank_score"],
                    Citation(
                        chunk_id=item["chunk"].chunk_id,
                        text=self._clean_citation(item["chunk"].text),
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

    def _bm25_scores(
        self,
        query_tokens: list[str],
        chunk_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[float]:
        if not query_tokens or not chunk_tokens:
            return [0 for _ in chunk_tokens]
        query = Counter(query_tokens)
        chunk_sets = [set(tokens) for tokens in chunk_tokens]
        document_count = len(chunk_tokens)
        average_length = sum(len(tokens) for tokens in chunk_tokens) / document_count or 1
        document_frequency = {
            token: sum(1 for tokens in chunk_sets if token in tokens) for token in query
        }

        scores = []
        for tokens in chunk_tokens:
            token_counts = Counter(tokens)
            document_length = len(tokens) or 1
            score = 0.0
            for token, query_count in query.items():
                frequency = token_counts[token]
                if frequency == 0:
                    continue
                df = document_frequency[token]
                idf = math.log(1 + ((document_count - df + 0.5) / (df + 0.5)))
                denominator = frequency + k1 * (
                    1 - b + b * (document_length / average_length)
                )
                score += query_count * idf * ((frequency * (k1 + 1)) / denominator)
            scores.append(score)
        return scores

    @classmethod
    def _embedding_vector(cls, tokens: list[str]) -> Counter[str]:
        vector: Counter[str] = Counter()
        for token in tokens:
            vector[token] += 1.0
            if len(token) >= 4:
                vector[f"prefix:{token[:4]}"] += 0.2
            for group_name, group_tokens in SEMANTIC_GROUPS.items():
                if token in group_tokens:
                    vector[f"semantic:{group_name}"] += 1.5
        return vector

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0
        intersection = set(left).intersection(right)
        numerator = sum(left[key] * right[key] for key in intersection)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        if left_norm == 0 or right_norm == 0:
            return 0
        return numerator / (left_norm * right_norm)

    @staticmethod
    def _rerank_score(
        final_score: float,
        query_tokens: list[str],
        chunk_tokens: list[str],
    ) -> float:
        query_set = set(query_tokens)
        if not query_set:
            return final_score
        coverage = len(query_set.intersection(chunk_tokens)) / len(query_set)
        return final_score + (0.1 * coverage)

    def _initial_top_k(self, document_count: int) -> int:
        return max(self.top_k_citations, self.top_k_citations * max(1, document_count) * 2)

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
