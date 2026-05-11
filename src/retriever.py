"""
Retriever — three retrieval strategies sharing a common interface.

Strategy A — Raw vector search
    Embed the query as-is and search the vector store.

Strategy B — Synonym-expanded search (mocked GenerativeModel)
    Pass the query through GenerativeModel.generate_content() which
    expands it with WordNet synonyms, then embed the expansion.

Strategy C — Reciprocal Rank Fusion
    Run A and B independently and merge the ranked lists with RRF.
    Rewards chunks that both strategies agree on.

All strategies return list[ScoredChunk] so the evaluator and
benchmark runner treat them identically.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.embedder import TextEmbeddingModel
from src.fusion import reciprocal_rank_fusion
from src.query_expander import GenerativeModel
from src.vector_store import ScoredChunk, VectorStore


@dataclass
class RetrievalResult:
    strategy: str
    query: str
    expanded_query: str | None
    chunks: list[ScoredChunk]


class StrategyA:
    """Raw vector search — no query modification."""

    name = "A — Raw vector search"

    def __init__(self, store: VectorStore, embedder: TextEmbeddingModel) -> None:
        self._store = store
        self._embedder = embedder

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        vec = self._embedder.embed_one(query)
        chunks = self._store.search(vec, top_k=top_k)
        return RetrievalResult(
            strategy=self.name,
            query=query,
            expanded_query=None,
            chunks=chunks,
        )


class StrategyB:
    """Synonym-expanded search via mocked GenerativeModel."""

    name = "B — Synonym expansion (mocked LLM)"

    def __init__(
        self,
        store: VectorStore,
        embedder: TextEmbeddingModel,
        expander: GenerativeModel | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._expander = expander or GenerativeModel()

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        response = self._expander.generate_content(query)
        expanded = response.text
        vec = self._embedder.embed_one(expanded)
        chunks = self._store.search(vec, top_k=top_k)
        return RetrievalResult(
            strategy=self.name,
            query=query,
            expanded_query=expanded,
            chunks=chunks,
        )


class StrategyC:
    """Reciprocal Rank Fusion — merges results from A and B."""

    name = "C — RRF (A ∪ B)"

    def __init__(
        self,
        store: VectorStore,
        embedder: TextEmbeddingModel,
        expander: GenerativeModel | None = None,
    ) -> None:
        self._a = StrategyA(store, embedder)
        self._b = StrategyB(store, embedder, expander)

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        result_a = self._a.retrieve(query, top_k=top_k)
        result_b = self._b.retrieve(query, top_k=top_k)
        merged = reciprocal_rank_fusion(
            [result_a.chunks, result_b.chunks],
            top_k=top_k,
        )
        return RetrievalResult(
            strategy=self.name,
            query=query,
            expanded_query=result_b.expanded_query,
            chunks=merged,
        )