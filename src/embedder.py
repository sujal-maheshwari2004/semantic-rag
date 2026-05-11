"""
Embedder, wraps sentence-transformers locally while exposing the
same interface as vertexai.language_models.TextEmbeddingModel so
the production swap is a one-line import change.

Vertex AI interface being mimicked:
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    embeddings = model.get_embeddings(texts)          # list[TextEmbedding]
    vector = embeddings[0].values                     # list[float]
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.cache import EmbeddingCache


class TextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding."""

    def __init__(self, values: list[float]) -> None:
        self.values = values


class TextEmbeddingModel:
    """
    Local stand-in for vertexai.language_models.TextEmbeddingModel.
    Uses all-MiniLM-L6-v2 (384-dim) to simulate textembedding-gecko
    behaviour.  Vectors are L2-normalised so inner product == cosine
    similarity — matching the behaviour of the Vertex AI model.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, cache_dir: str = ".cache") -> None:
        self._model = SentenceTransformer(self._MODEL_NAME)
        self._cache = EmbeddingCache(cache_dir)

    @classmethod
    def from_pretrained(cls, model_name: str = "textembedding-gecko@003") -> "TextEmbeddingModel":
        """Class-method mirrors the Vertex AI SDK entry point."""
        return cls()

    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        """
        Returns a list of TextEmbedding objects — one per input text.
        Checks the cache first; only calls the model for cache misses.
        """
        results: list[TextEmbedding] = []
        to_encode: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(TextEmbedding(cached.tolist()))
            else:
                results.append(None)  # type: ignore[arg-type]
                to_encode.append((i, text))

        if to_encode:
            indices, raw_texts = zip(*to_encode)
            vectors = self._model.encode(list(raw_texts), normalize_embeddings=True)
            for idx, vec in zip(indices, vectors):
                self._cache.set(texts[idx], vec)
                results[idx] = TextEmbedding(vec.tolist())

        return results

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience method — returns a normalised numpy vector."""
        return np.array(self.get_embeddings([text])[0].values, dtype=np.float32)

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Returns shape (N, D) normalised matrix."""
        embeddings = self.get_embeddings(texts)
        return np.array([e.values for e in embeddings], dtype=np.float32)