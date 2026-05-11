"""
Embedding cache —> sha256(text) to np.ndarray.
Backed by a .npy array + .json index on disk.
In production this would be Cloud Memorystore or a
Vertex AI embedding endpoint response cache.

Thread-safety:
    A threading.Lock guards all in-memory mutations.
    Disk writes use atomic rename (write to .tmp, then os.replace)
    so a crash mid-write never leaves a corrupt cache file.
    Concurrent readers are safe because numpy load is read-only and
    the index dict is replaced atomically on load.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(exist_ok=True)
        self._index_path = self._dir / "index.json"
        self._vectors_path = self._dir / "vectors.npy"
        self._lock = threading.Lock()
        self._index: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._load()

    # ------------------------------------------------------------------
    # public api
    # ------------------------------------------------------------------

    def get(self, text: str) -> np.ndarray | None:
        key = self._hash(text)
        with self._lock:
            if key not in self._index:
                return None
            return self._vectors[self._index[key]]

    def set(self, text: str, embedding: np.ndarray) -> None:
        key = self._hash(text)
        with self._lock:
            if key in self._index:
                return
            self._index[key] = len(self._vectors)
            self._vectors.append(embedding)
            self._persist()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _load(self) -> None:
        """Load index and vectors from disk. Called once at init (no lock needed)."""
        if self._index_path.exists():
            self._index = json.loads(self._index_path.read_text())
        if self._vectors_path.exists():
            arr = np.load(str(self._vectors_path))
            self._vectors = [arr[i] for i in range(len(arr))]

    def _persist(self) -> None:
        """
        Atomically flush index and vectors to disk.
        Must be called with self._lock held.

        Strategy:
          1. Write to a sibling .tmp file.
          2. os.replace() — atomic on POSIX, atomic on Windows (Python 3.3+).
          This guarantees readers never see a half-written file.
        """
        # --- vectors ---
        vec_tmp = str(self._vectors_path) + ".tmp"
        np.save(vec_tmp, np.array(self._vectors))
        os.replace(vec_tmp, str(self._vectors_path))

        # --- index ---
        idx_tmp = str(self._index_path) + ".tmp"
        Path(idx_tmp).write_text(json.dumps(self._index))
        os.replace(idx_tmp, str(self._index_path))