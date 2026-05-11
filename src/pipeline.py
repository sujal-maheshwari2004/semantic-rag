"""
Pipeline — orchestrates ingestion and retrieval.

RAGPipeline is the single entry point for:
  - ingest(corpus)  : embed chunks and populate the vector store
  - query(...)      : delegate to the requested strategy

Nothing outside this module needs to know about the embedder,
store, or retriever implementations.
"""

from __future__ import annotations

from src.embedder import TextEmbeddingModel
from src.query_expander import GenerativeModel
from src.retriever import RetrievalResult, StrategyA, StrategyB, StrategyC
from src.vector_store import VectorStore

STRATEGY_MAP = {"A": StrategyA, "B": StrategyB, "C": StrategyC}


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

    def ingest(self, corpus: list[dict]) -> None:
        """
        corpus: list of dicts with 'id' and 'text' keys.
        Embeds all chunks and loads them into the vector store.
        """
        texts = [c["text"] for c in corpus]
        embeddings = self._embedder.embed_many(texts)
        self._store.add(corpus, embeddings)
        self._ingested = True
        print(f"[pipeline] ingested {len(corpus)} chunks → store size={self._store.size}")

    def query(
        self,
        query: str,
        strategy: str = "A",
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        strategy: 'A' | 'B' | 'C'
        """
        if not self._ingested:
            raise RuntimeError("Call ingest() before query().")
        if strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose A, B, or C.")

        retriever_cls = STRATEGY_MAP[strategy]

        if strategy == "A":
            retriever = retriever_cls(self._store, self._embedder)
        else:
            retriever = retriever_cls(self._store, self._embedder, self._expander)

        return retriever.retrieve(query, top_k=top_k)

    @property
    def store(self) -> VectorStore:
        return self._store