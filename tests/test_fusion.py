"""Tests for Reciprocal Rank Fusion."""

import pytest

from src.fusion import reciprocal_rank_fusion
from src.vector_store import ScoredChunk


def make_chunks(ids_with_ranks: list[tuple[str, int]]) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk_id=cid, text=f"text {cid}", score=1.0, rank=rank)
        for cid, rank in ids_with_ranks
    ]


def test_rrf_returns_top_k():
    list_a = make_chunks([("c1", 1), ("c2", 2), ("c3", 3)])
    list_b = make_chunks([("c2", 1), ("c1", 2), ("c4", 3)])
    result = reciprocal_rank_fusion([list_a, list_b], top_k=2)
    assert len(result) == 2


def test_rrf_agreement_boosts_rank():
    """A chunk appearing rank-1 in both lists should win."""
    list_a = make_chunks([("winner", 1), ("c2", 2), ("c3", 3)])
    list_b = make_chunks([("winner", 1), ("c4", 2), ("c5", 3)])
    result = reciprocal_rank_fusion([list_a, list_b], top_k=3)
    assert result[0].chunk_id == "winner"


def test_rrf_scores_descending():
    list_a = make_chunks([("c1", 1), ("c2", 2), ("c3", 3)])
    list_b = make_chunks([("c3", 1), ("c1", 2), ("c2", 3)])
    result = reciprocal_rank_fusion([list_a, list_b], top_k=3)
    scores = [r.score for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_ranks_are_1_indexed():
    list_a = make_chunks([("c1", 1), ("c2", 2)])
    list_b = make_chunks([("c1", 1), ("c2", 2)])
    result = reciprocal_rank_fusion([list_a, list_b], top_k=2)
    assert [r.rank for r in result] == [1, 2]


def test_rrf_deduplicates_chunks():
    list_a = make_chunks([("c1", 1), ("c2", 2), ("c3", 3)])
    list_b = make_chunks([("c1", 1), ("c2", 2), ("c3", 3)])
    result = reciprocal_rank_fusion([list_a, list_b], top_k=5)
    ids = [r.chunk_id for r in result]
    assert len(ids) == len(set(ids))


def test_rrf_single_list_passthrough():
    list_a = make_chunks([("c1", 1), ("c2", 2), ("c3", 3)])
    result = reciprocal_rank_fusion([list_a], top_k=3)
    assert [r.chunk_id for r in result] == ["c1", "c2", "c3"]


def test_rrf_score_formula():
    """Verify the RRF score for a rank-1 chunk with k=60."""
    list_a = make_chunks([("c1", 1)])
    result = reciprocal_rank_fusion([list_a], k=60, top_k=1)
    expected = round(1 / (60 + 1), 6)
    assert result[0].score == expected