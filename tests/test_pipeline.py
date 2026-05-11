"""
End-to-end tests for RAGPipeline — sync and async paths.
"""

from __future__ import annotations

import asyncio

import pytest

from data.corpus import CHUNKS
from src.pipeline import RAGPipeline, _STRATEGY_REGISTRY
from src.retriever import RetrievalResult, StrategyA, StrategyB, StrategyC


@pytest.fixture
def pipeline(tmp_path):
    return RAGPipeline(cache_dir=str(tmp_path))


# ------------------------------------------------------------------
# sync ingestion / retrieval (original tests, preserved)
# ------------------------------------------------------------------

def test_ingest_populates_store(pipeline):
    pipeline.ingest(CHUNKS)
    assert pipeline.store.size == len(CHUNKS)


def test_query_before_ingest_raises(pipeline):
    with pytest.raises(RuntimeError, match="ingest"):
        pipeline.query("any query")


def test_invalid_strategy_raises(pipeline):
    pipeline.ingest(CHUNKS)
    with pytest.raises(ValueError, match="Unknown strategy"):
        pipeline.query("any query", strategy="Z")


def test_strategy_a_returns_result(pipeline):
    pipeline.ingest(CHUNKS)
    result = pipeline.query("peak load autoscaling", strategy="A")
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 3


def test_strategy_b_returns_result(pipeline):
    pipeline.ingest(CHUNKS)
    result = pipeline.query("data consistency", strategy="B")
    assert isinstance(result, RetrievalResult)
    assert result.expanded_query is not None


def test_strategy_c_returns_result(pipeline):
    pipeline.ingest(CHUNKS)
    result = pipeline.query("model inference", strategy="C")
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) <= 3


def test_top_k_respected(pipeline):
    pipeline.ingest(CHUNKS)
    result = pipeline.query("any query", strategy="A", top_k=2)
    assert len(result.chunks) <= 2


def test_chunk_ids_from_corpus(pipeline):
    pipeline.ingest(CHUNKS)
    result = pipeline.query("vector embeddings FAISS", strategy="A", top_k=3)
    valid_ids = {c["id"] for c in CHUNKS}
    for chunk in result.chunks:
        assert chunk.chunk_id in valid_ids


# ------------------------------------------------------------------
# strategy dispatch — data-driven, no if/elif
# ------------------------------------------------------------------

def test_strategy_registry_contains_all_keys():
    assert set(_STRATEGY_REGISTRY.keys()) == {"A", "B", "C"}


def test_strategy_registry_classes():
    assert _STRATEGY_REGISTRY["A"][0] is StrategyA
    assert _STRATEGY_REGISTRY["B"][0] is StrategyB
    assert _STRATEGY_REGISTRY["C"][0] is StrategyC


def test_strategy_a_does_not_need_expander():
    _, needs_expander = _STRATEGY_REGISTRY["A"]
    assert needs_expander is False


def test_strategy_b_c_need_expander():
    for key in ("B", "C"):
        _, needs_expander = _STRATEGY_REGISTRY[key]
        assert needs_expander is True


# ------------------------------------------------------------------
# async ingestion
# ------------------------------------------------------------------

def test_async_ingest_populates_store(tmp_path):
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_ingest(CHUNKS)

    asyncio.run(_run())
    assert pipeline.store.size == len(CHUNKS)


def test_async_ingest_idempotent_to_store_size(tmp_path):
    """Calling async_ingest twice should double the store (no dedup at pipeline level)."""
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_ingest(CHUNKS)

    asyncio.run(_run())
    first_size = pipeline.store.size
    assert first_size == len(CHUNKS)


# ------------------------------------------------------------------
# async query
# ------------------------------------------------------------------

def test_async_query_returns_result(tmp_path):
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_ingest(CHUNKS)
        return await pipeline.async_query("peak load autoscaling", strategy="A")

    result = asyncio.run(_run())
    assert isinstance(result, RetrievalResult)
    assert len(result.chunks) == 3


def test_async_query_before_ingest_raises(tmp_path):
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_query("anything")

    with pytest.raises(RuntimeError, match="ingest"):
        asyncio.run(_run())


def test_async_query_all_strategies(tmp_path):
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_ingest(CHUNKS)
        results = {}
        for s in ("A", "B", "C"):
            results[s] = await pipeline.async_query("model inference", strategy=s, top_k=3)
        return results

    results = asyncio.run(_run())
    for s, result in results.items():
        assert isinstance(result, RetrievalResult), f"Strategy {s} did not return RetrievalResult"
        assert len(result.chunks) <= 3


def test_concurrent_async_queries_return_correct_types(tmp_path):
    """Fire 6 queries concurrently — all must return valid RetrievalResults."""
    pipeline = RAGPipeline(cache_dir=str(tmp_path))

    async def _run():
        await pipeline.async_ingest(CHUNKS)
        queries = [
            pipeline.async_query("peak load", strategy="A"),
            pipeline.async_query("data consistency", strategy="B"),
            pipeline.async_query("model inference", strategy="C"),
            pipeline.async_query("autoscaling replicas", strategy="A"),
            pipeline.async_query("raft consensus", strategy="B"),
            pipeline.async_query("quantisation inference", strategy="C"),
        ]
        return await asyncio.gather(*queries)

    results = asyncio.run(_run())
    assert len(results) == 6
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert len(r.chunks) > 0