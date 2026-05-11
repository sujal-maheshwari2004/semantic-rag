"""
Pipeline — orchestrates ingestion and retrieval.

RAGPipeline is the single entry point for:
  - ingest(corpus)        : embed chunks and populate the vector store
  - async_ingest(corpus)  : async wrapper — awaitable from async callers
  - query(...)            : delegate to the requested strategy (sync)
  - async_query(...)      : async wrapper — awaitable from async callers

Async design:
    The CPU-bound work (sentence-transformer encode + FAISS search) runs
    in a thread pool via asyncio.get_event_loop().run_in_executor so the
    event loop is never blocked.  This matches the pattern used by
    production async serving frameworks (FastAPI, aiohttp) where
    multiple concurrent requests share a single event loop.

    For IO-bound production backends (Vertex AI Matching Engine,
    Cloud Memorystore) the executor call would be replaced by a native
    async client (aiohttp / google-cloud-aiplatform async stubs).

Strategy dispatch:
    All three strategies are instantiated uniformly through STRATEGY_MAP.
    StrategyA accepts (store, embedder); B and C accept
    (store, embedder, expander).  The map stores (cls, needs_expander)
    tuples so dispatch is data-driven with no if/elif branching.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.embedder import TextEmbeddingModel
from src.query_expander import GenerativeModel
from src.retriever import RetrievalResult, StrategyA, StrategyB, StrategyC
from src.vector_store import VectorStore

# (strategy_class, requires_expander)
_STRATEGY_REGISTRY: dict[str, tuple[Any, bool]] = {
    "A": (StrategyA, False),
    "B": (StrategyB, True),
    "C": (StrategyC, True),
}

# Shared executor for offloading CPU-bound work from the event loop.
# Max workers kept small — sentence-transformer is already multi-threaded
# internally via BLAS/OpenMP; over-parallelising causes contention.
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


class RAGPipeline:
    def __init__(
        self,
        cache_dir: str = ".cache",
        expander: GenerativeModel | None = None,
    ) -> None:
        self._embedder = TextEmbeddingModel(cache_dir=cache_dir)
        self._store = VectorStore(dim=384)
        self._expander = expander or GenerativeModel()
        self._ingested = False

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def ingest(self, corpus: list[dict]) -> None:
        """
        Synchronous ingestion.
        corpus: list of dicts with 'id' and 'text' keys.
        """
        texts = [c["text"] for c in corpus]
        embeddings = self._embedder.embed_many(texts)
        self._store.add(corpus, embeddings)
        self._ingested = True
        print(f"[pipeline] ingested {len(corpus)} chunks → store size={self._store.size}")

    async def async_ingest(self, corpus: list[dict]) -> None:
        """
        Async wrapper around ingest() — offloads embedding to thread pool
        so the event loop stays free for other coroutines.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_EXECUTOR, self.ingest, corpus)

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def query(
        self,
        query: str,
        strategy: str = "A",
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Synchronous retrieval.
        strategy: 'A' | 'B' | 'C'
        """
        if not self._ingested:
            raise RuntimeError("Call ingest() before query().")
        if strategy not in _STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose A, B, or C.")

        retriever_cls, needs_expander = _STRATEGY_REGISTRY[strategy]

        if needs_expander:
            retriever = retriever_cls(self._store, self._embedder, self._expander)
        else:
            retriever = retriever_cls(self._store, self._embedder)

        return retriever.retrieve(query, top_k=top_k)

    async def async_query(
        self,
        query: str,
        strategy: str = "A",
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        Async wrapper around query() — offloads FAISS search to thread pool.
        Suitable for use inside FastAPI route handlers or other async contexts.

        Example::

            pipeline = RAGPipeline()
            await pipeline.async_ingest(CHUNKS)
            result = await pipeline.async_query("peak load", strategy="C")
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _EXECUTOR,
            lambda: self.query(query, strategy=strategy, top_k=top_k),
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> VectorStore:
        return self._store