"""
Thread-safety tests for EmbeddingCache.

These tests simulate the concurrent write pattern that occurs when the
embedder processes a large batch using multiple threads.  The key
invariants are:

1. No data is lost — every set() call is reflected in a subsequent get().
2. The on-disk files are never corrupt — index and vectors stay consistent.
3. __len__ is accurate after concurrent writes.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from src.cache import EmbeddingCache


def _make_vec(seed: int, dim: int = 384) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


# ------------------------------------------------------------------
# basic correctness (preserved from original suite)
# ------------------------------------------------------------------

def test_set_and_get_roundtrip(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    vec = _make_vec(0)
    cache.set("hello", vec)
    result = cache.get("hello")
    assert result is not None
    np.testing.assert_array_almost_equal(result, vec)


def test_get_missing_returns_none(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    assert cache.get("nonexistent") is None


def test_set_is_idempotent(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    vec = _make_vec(1)
    cache.set("key", vec)
    cache.set("key", _make_vec(2))   # second write for same key is ignored
    assert len(cache) == 1


def test_len_reflects_entries(tmp_path):
    cache = EmbeddingCache(str(tmp_path))
    for i in range(5):
        cache.set(f"text_{i}", _make_vec(i))
    assert len(cache) == 5


def test_persists_across_instances(tmp_path):
    vec = _make_vec(99)
    c1 = EmbeddingCache(str(tmp_path))
    c1.set("persistent", vec)

    c2 = EmbeddingCache(str(tmp_path))
    result = c2.get("persistent")
    assert result is not None
    np.testing.assert_array_almost_equal(result, vec)


# ------------------------------------------------------------------
# thread-safety
# ------------------------------------------------------------------

def test_concurrent_writes_no_data_loss(tmp_path):
    """
    100 threads each write a unique key concurrently.
    After all threads finish, every key must be retrievable.
    """
    cache = EmbeddingCache(str(tmp_path))
    n = 100
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            cache.set(f"concurrent_key_{i}", _make_vec(i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions in threads: {errors}"
    assert len(cache) == n
    for i in range(n):
        assert cache.get(f"concurrent_key_{i}") is not None, f"key {i} missing"


def test_concurrent_writes_correct_vectors(tmp_path):
    """
    Each thread writes a deterministic vector; values retrieved after
    all threads complete must match what was written.
    """
    cache = EmbeddingCache(str(tmp_path))
    n = 50
    written: dict[str, np.ndarray] = {f"key_{i}": _make_vec(i) for i in range(n)}

    def worker(key: str, vec: np.ndarray) -> None:
        cache.set(key, vec)

    threads = [threading.Thread(target=worker, args=(k, v)) for k, v in written.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for key, expected in written.items():
        retrieved = cache.get(key)
        assert retrieved is not None
        np.testing.assert_array_almost_equal(retrieved, expected, decimal=5)


def test_concurrent_reads_while_writing(tmp_path):
    """
    Writer threads and reader threads run simultaneously.
    Readers must never raise an exception (they may return None for
    keys not yet written, which is acceptable).
    """
    cache = EmbeddingCache(str(tmp_path))
    # Pre-populate some keys so readers have something to find.
    for i in range(10):
        cache.set(f"pre_{i}", _make_vec(i))

    read_errors: list[Exception] = []
    write_errors: list[Exception] = []

    def reader() -> None:
        for i in range(50):
            try:
                cache.get(f"pre_{i % 10}")
            except Exception as exc:
                read_errors.append(exc)

    def writer(start: int) -> None:
        for i in range(start, start + 10):
            try:
                cache.set(f"new_{i}", _make_vec(i + 1000))
            except Exception as exc:
                write_errors.append(exc)

    threads = (
        [threading.Thread(target=reader) for _ in range(5)]
        + [threading.Thread(target=writer, args=(i * 10,)) for i in range(5)]
    )
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not read_errors, f"Read exceptions: {read_errors}"
    assert not write_errors, f"Write exceptions: {write_errors}"


def test_atomic_write_no_corrupt_index(tmp_path):
    """
    After many concurrent writes, reloading the cache from disk must
    produce a consistent index — no KeyError, no shape mismatch.
    """
    cache = EmbeddingCache(str(tmp_path))
    n = 60

    threads = [
        threading.Thread(target=cache.set, args=(f"t_{i}", _make_vec(i)))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Reload from disk and verify structural integrity.
    reloaded = EmbeddingCache(str(tmp_path))
    assert len(reloaded) == n
    for i in range(n):
        v = reloaded.get(f"t_{i}")
        assert v is not None
        assert v.shape == (384,)