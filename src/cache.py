"""
Embedding cache —> sha256(text) to np.ndarray.
Backed by a .npy array + .json index on disk.
In production this would be Cloud Memorystore or a
Vertex AI embedding endpoint response cache.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._vectors_path = self._dir / "vectors.npy"
        self._index: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._load()

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def get(self, text: str) -> np.ndarray | None:
        key = self._hash(text)
        if key not in self._index:
            return None
        row = self._index[key]
        return self._vectors[row]

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._hash(text)
        if key in self._index:
            return
        self._index[key] = len(self._vectors)
        self._vectors.append(embedding)
        self._persist()

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load(self) -> None:
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())
        if self._vectors_path.exists():
            arr = np.load(str(self._vectors_path))
            self._vectors = [arr[i] for i in range(len(arr))]

    def _persist(self) -> None:
        self._index_path.write_text(json.dumps(self._index))
        np.save(str(self._vectors_path), np.array(self._vectors))