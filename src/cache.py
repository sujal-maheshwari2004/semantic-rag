"""
Embedding cache -> sha256(text) to np.ndarray.
Backed by a .npy array + .json index on disk.
In production this would be Cloud Memorystore or a
Vertex AI embedding endpoint response cache.

Thread-safety:
    A threading.Lock guards all in-memory mutations.
    Disk writes use atomic rename (write to .tmp, then os.replace)
    so a crash mid-write never leaves a corrupt cache file.

Windows note:
    numpy.save(path) silently appends '.npy' to paths that don't
    already end in '.npy', so writing to 'vectors.npy.tmp' produces
    'vectors.npy.tmp.npy' and os.replace() cannot find the source.
    Fix: serialise into a BytesIO buffer, then write raw bytes via
    Path.write_bytes() — full control of the filename, cross-platform.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, cache_dir: str = ".cache") -> None:
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
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

        numpy.save(path) appends '.npy' when the path doesn't already end
        in '.npy', so 'vectors.npy.tmp' becomes 'vectors.npy.tmp.npy' and
        os.replace() fails on Windows.  We serialise into a BytesIO buffer
        and write raw bytes ourselves to avoid that behaviour.
        """
        # --- vectors: BytesIO avoids numpy's implicit .npy suffix ---
        buf = io.BytesIO()
        np.save(buf, np.array(self._vectors))
        vec_tmp = self._dir / "vectors.npy.tmp"
        vec_tmp.write_bytes(buf.getvalue())
        os.replace(str(vec_tmp), str(self._vectors_path))

        # --- index ---
        idx_tmp = self._dir / "index.json.tmp"
        idx_tmp.write_text(json.dumps(self._index), encoding="utf-8")
        os.replace(str(idx_tmp), str(self._index_path))