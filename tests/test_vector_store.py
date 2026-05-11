"""Tests for VectorStore."""

import numpy as np
import pytest

from src.vector_store import ScoredChunk, VectorStore


def make_store(n: int = 5, dim: int = 384) -> tuple[VectorStore, list[dict], np.ndarray]:
    store = VectorStore(dim=dim)
    chunks = [{"id": f"chunk_{i:02d}", "text": f"text {i}"} for i in range(n)]
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    return store, chunks, vecs


def test_add_increases_store_size():
    store, chunks, vecs = make_store(5)
    store.add(chunks, vecs)
    assert store.size == 5


def test_search_returns_scored_chunks():
    store, chunks, vecs = make_store(5)
    store.add(chunks, vecs)
    results = store.search(vecs[0], top_k=3)
    assert all(isinstance(r, ScoredChunk) for r in results)


def test_search_top_k_limit():
    store, chunks, vecs = make_store(5)
    store.add(chunks, vecs)
    results = store.search(vecs[0], top_k=2)
    assert len(results) == 2


def test_search_first_result_is_self(make_normalised_vec=None):
    store, chunks, vecs = make_store(5)
    store.add(chunks, vecs)
    results = store.search(vecs[2], top_k=1)
    assert results[0].chunk_id == "chunk_02"


def test_search_scores_descending():
    store, chunks, vecs = make_store(5)
    store.add(chunks, vecs)
    results = store.search(vecs[0], top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_on_empty_store_returns_empty():
    store = VectorStore(dim=384)
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    assert store.search(query) == []


def test_add_wrong_dim_raises():
    store = VectorStore(dim=384)
    bad = np.ones((2, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        store.add([{"id": "x", "text": "y"}], bad)


def test_rank_field_is_1_indexed():
    store, chunks, vecs = make_store(3)
    store.add(chunks, vecs)
    results = store.search(vecs[0], top_k=3)
    ranks = [r.rank for r in results]
    assert ranks == [1, 2, 3]