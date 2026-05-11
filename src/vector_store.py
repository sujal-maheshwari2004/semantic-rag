"""
Vector store is a FAISS IndexFlatIP backed store.

IndexFlatIP performs exact inner-product search.  Because all vectors
are L2-normalised by the embedder, inner product == cosine similarity.
This matches the default metric used by Vertex AI Matching Engine.

Production migration path:
    Replace search() with a call to
    index_endpoint.find_neighbors(queries, num_neighbors=top_k)
    and map the MatchNeighbor response to ScoredChunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class ScoredChunk:
    chunk_id: str
    text: str
    score: float
    rank: int = 0


class VectorStore:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[dict] = []

    # ------------------------------------------------------------------
    # write path
    # ------------------------------------------------------------------

    def add(self, chunks: list[dict], embeddings: np.ndarray) -> None:
        """
        chunks   : list of dicts with keys 'id' and 'text'
        embeddings: float32 array of shape (N, dim), already L2-normalised
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Expected embeddings of shape (N, {self._dim}), "
                f"got {embeddings.shape}"
            )
        self._index.add(embeddings)
        self._chunks.extend(chunks)

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> list[ScoredChunk]:
        """
        query_embedding: 1-D or 2-D float32 array, L2-normalised.
        Returns top_k ScoredChunks sorted by descending cosine similarity.
        """
        if self._index.ntotal == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(vec, min(top_k, self._index.ntotal))

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append(
                ScoredChunk(
                    chunk_id=chunk["id"],
                    text=chunk["text"],
                    score=float(score),
                    rank=rank + 1,
                )
            )
        return results

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    def load(self, path: str) -> None:
        self._index = faiss.read_index(path)

    @property
    def size(self) -> int:
        return self._index.ntotal