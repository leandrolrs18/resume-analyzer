import math
import logging
import os
import re
from collections import Counter
from typing import Any

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.schemas import Citation, RankingEvidence, ResumeDocument

BM25_WEIGHT = 0.3
EMBEDDING_WEIGHT = 0.7

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9+#.]+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?\d?\s?\d{4}[-\s]?\d{4}")
URL_RE = re.compile(r"https?://\S+")

# Sem tabelas de expansão estáticas para manter o código limpo e previsível.
logger = logging.getLogger(__name__)


class RankingService:
    def __init__(
        self,
        top_k_citations: int = 3,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        retrieval_top_k: int = 25,
        rerank_top_n: int = 5,
    ):
        self.top_k_citations = top_k_citations
        self.embedding_model_name = embedding_model_name
        self.rerank_model_name = rerank_model_name
        self.retrieval_top_k = retrieval_top_k
        self.rerank_top_n = rerank_top_n
        self.enable_cross_encoder = (
            os.environ.get("USE_CROSS_ENCODER_RERANK", "true").lower() == "true"
        )
        self._embedding_model: Any | None = None
        self._rerank_model: Any | None = None

    async def rank(
        self,
        query: str,
        documents: list[ResumeDocument],
    ) -> list[RankingEvidence]:
        # Fluxo de ranking:
        # 1) Recebe a query e os currículos já divididos em chunks.
        # 2) Tokeniza query e chunks.
        # 3) Faz a busca inicial com BM25 + SentenceTransformer.
        # 4) Reordena os melhores chunks com CrossEncoder.
        # 5) Agrupa por candidato e devolve evidências com citações.
        query_tokens = self._tokens(query)
        chunk_items = [
            (document, chunk, self._tokens(chunk.text))
            for document in documents
            for chunk in document.chunks
        ]
        # _bm25_scores:
        # 1) Recebe tokens da query e tokens de todos os chunks.
        # 2) Compara termos exatos, frequência e raridade dos termos.
        # 3) Retorna list[float], um score lexical para cada chunk.
        # 4) Exemplo: [1.42, 0.0, 0.83].
        bm25_scores = self._bm25_scores(query_tokens, [tokens for _, _, tokens in chunk_items])
        # _semantic_scores:
        # 1) Recebe a query em texto e todos os chunks em texto.
        # 2) Usa SentenceTransformer para gerar embeddings reais.
        # 3) Calcula similaridade cosseno entre query e cada chunk.
        # 4) Retorna list[float], um score semântico para cada chunk.
        semantic_scores = self._semantic_scores(
            query,
            [chunk.text for _, chunk, _ in chunk_items],
        )

        initial_candidates = []
        for (document, chunk, chunk_tokens), bm25_raw, semantic_score in zip(
            chunk_items, bm25_scores, semantic_scores, strict=False
        ):
            # BM25 preserva termos exatos; embedding real captura semântica/contexto.
            bm25_score = bm25_raw / (max(bm25_scores, default=0) or 1)
            final_score = (EMBEDDING_WEIGHT * semantic_score) + (BM25_WEIGHT * bm25_score)
            item = {
                "candidate": document.candidate,
                "chunk": chunk,
                "chunk_tokens": chunk_tokens,
                "bm25_raw": bm25_raw,
                "bm25_score": bm25_score,
                "embedding_score": semantic_score,
                "final_score": final_score,
            }
            initial_candidates.append(item)

        top_k_initial = sorted(
            (item for item in initial_candidates if item["final_score"] > 0),
            reverse=True,
            key=lambda item: item["final_score"],
        )[: self._initial_top_k(len(documents))]

        # Rerank clássico: query + chunk passam juntos pelo CrossEncoder.
        rerank_scores = self._rerank_scores(
            query,
            [item["chunk"].text for item in top_k_initial],
            [item["final_score"] for item in top_k_initial],
        )
        reranked = sorted(
            (
                {
                    **item,
                    "rerank_score": rerank_score,
                }
                for item, rerank_score in zip(top_k_initial, rerank_scores, strict=False)
            ),
            reverse=True,
            key=lambda item: item["rerank_score"],
        )[: self._rerank_top_n(len(documents))]

        scored: dict[str, list[tuple[float, Citation]]] = {}
        for item in reranked:
            # Cada chunk relevante vira uma Citation redigida para não expor contato/link.
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
        # Agrupa as melhores citações por candidato e normaliza o score final entre 0 e 1.
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
        # BM25 lexical puro: mede o quanto os termos da query aparecem nos chunks.
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
                denominator = frequency + k1 * (1 - b + b * (document_length / average_length))
                score += query_count * idf * ((frequency * (k1 + 1)) / denominator)
            scores.append(score)
        return scores

    def _semantic_scores(self, query: str, chunks: list[str]) -> list[float]:
        # Embedding real: SentenceTransformer transforma query/chunks em vetores semânticos.
        if not chunks:
            return []
        try:
            model = self._get_embedding_model()
            query_embedding = model.encode(query, normalize_embeddings=True)
            chunk_embeddings = model.encode(chunks, normalize_embeddings=True)
            return [
                float(query_embedding @ chunk_embedding) for chunk_embedding in chunk_embeddings
            ]
        except Exception:
            logger.warning("sentence_transformer_embedding_unavailable", exc_info=True)
            return [0.0 for _ in chunks]

    def _rerank_scores(
        self,
        query: str,
        chunks: list[str],
        fallback_scores: list[float],
    ) -> list[float]:
        # Reranker clássico: CrossEncoder lê query + chunk juntos.
        # O modelo retorna um logit bruto, então convertemos para 0..1 antes de ranquear.
        if not chunks:
            return []
        if not self.enable_cross_encoder:
            return fallback_scores
        try:
            model = self._get_rerank_model()
            pairs = [(query, chunk) for chunk in chunks]
            raw_scores = [float(score) for score in model.predict(pairs)]
            normalized_scores = [self._sigmoid(score) for score in raw_scores]
            return normalized_scores
        except Exception:
            logger.warning("cross_encoder_rerank_unavailable", exc_info=True)
            return fallback_scores

    def _get_embedding_model(self) -> SentenceTransformer:
        # Carregamento lazy: usa modelo local/cacheado e evita download durante a request.
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device="cpu",
                local_files_only=True,
            )
        return self._embedding_model

    def _get_rerank_model(self) -> CrossEncoder:
        # Carregamento lazy: usa modelo local/cacheado e evita download durante a request.
        if self._rerank_model is None:
            self._rerank_model = CrossEncoder(
                self.rerank_model_name,
                device="cpu",
                local_files_only=True,
            )
        return self._rerank_model

    def _initial_top_k(self, document_count: int) -> int:
        # Mantém um conjunto inicial maior para o rerank quando há vários currículos.
        return max(self.retrieval_top_k, self.top_k_citations * max(1, document_count) * 2)

    def _rerank_top_n(self, document_count: int) -> int:
        # Seleciona apenas os melhores chunks após o rerank para a resposta final.
        return max(
            self.top_k_citations, min(self.rerank_top_n, self._initial_top_k(document_count))
        )

    @staticmethod
    def _tokens(text: str) -> list[str]:
        # Tokenização simples, case-insensitive, preservando tecnologias e siglas úteis.
        return [token.lower() for token in TOKEN_RE.findall(text)]

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1 / (1 + math.exp(-value))

    @staticmethod
    def _clean_citation(text: str) -> str:
        # Remove contato e links antes de retornar a citação.
        text = URL_RE.sub("[url]", text)
        text = EMAIL_RE.sub("[email]", text)
        text = PHONE_RE.sub("[telefone]", text)
        return " ".join(text.split())[:700].rstrip()
