"""
Tests for all three retrieval strategies.
GenerativeModel is patched so these tests are fully offline.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.embedder import TextEmbeddingModel
from src.retriever import RetrievalResult, StrategyA, StrategyB, StrategyC
from src.vector_store import VectorStore


@pytest.fixture
def populated_store():
    store = VectorStore(dim=384)
    chunks = [
        {"id": f"chunk_{i:02d}", "text": f"technical content about topic {i}"}
        for i in range(5)
    ]
    vecs = np.random.randn(5, 384).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    store.add(chunks, vecs)
    return store


@pytest.fixture
def embedder(tmp_path):
    return TextEmbeddingModel(cache_dir=str(tmp_path))


def test_strategy_a_returns_retrieval_result(populated_store, embedder):
    strategy = StrategyA(populated_store, embedder)
    result = strategy.retrieve("peak load handling", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.strategy == StrategyA.name
    assert len(result.chunks) == 3


def test_strategy_a_no_expanded_query(populated_store, embedder):
    strategy = StrategyA(populated_store, embedder)
    result = strategy.retrieve("any query")
    assert result.expanded_query is None


def test_strategy_b_expanded_query_set(populated_store, embedder):
    strategy = StrategyB(populated_store, embedder)
    result = strategy.retrieve("data consistency mechanisms")
    assert result.expanded_query is not None
    assert len(result.expanded_query) > len("data consistency mechanisms")


def test_strategy_b_uses_mock_expander(populated_store, embedder):
    mock_expander = MagicMock()
    mock_expander.generate_content.return_value = MagicMock(
        text="data consistency mechanisms quorum replication"
    )
    strategy = StrategyB(populated_store, embedder, expander=mock_expander)
    result = strategy.retrieve("data consistency mechanisms")
    mock_expander.generate_content.assert_called_once()
    assert result.expanded_query == "data consistency mechanisms quorum replication"


def test_strategy_c_returns_merged_results(populated_store, embedder):
    strategy = StrategyC(populated_store, embedder)
    result = strategy.retrieve("model inference optimisation", top_k=3)
    assert isinstance(result, RetrievalResult)
    assert result.strategy == StrategyC.name
    assert len(result.chunks) <= 3


def test_strategy_c_chunk_ids_are_unique(populated_store, embedder):
    strategy = StrategyC(populated_store, embedder)
    result = strategy.retrieve("any query", top_k=3)
    ids = [c.chunk_id for c in result.chunks]
    assert len(ids) == len(set(ids))


def test_all_strategies_return_same_interface(populated_store, embedder):
    query = "system scalability"
    for cls in [StrategyA, StrategyB, StrategyC]:
        s = cls(populated_store, embedder)
        result = s.retrieve(query, top_k=2)
        assert isinstance(result, RetrievalResult)
        assert result.query == query
        assert all(hasattr(c, "chunk_id") for c in result.chunks)