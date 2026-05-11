"""End-to-end tests for RAGPipeline."""

import pytest

from data.corpus import CHUNKS
from src.pipeline import RAGPipeline
from src.retriever import RetrievalResult


@pytest.fixture
def pipeline(tmp_path):
    return RAGPipeline(cache_dir=str(tmp_path))


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