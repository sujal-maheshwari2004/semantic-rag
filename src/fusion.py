"""
Fusion — Reciprocal Rank Fusion (RRF) over multiple ranked lists.

RRF score formula (Cormack et al. 2009):
    score(chunk) = Σ  1 / (k + rank_i)
                  lists

k=60 is the standard constant that dampens the outsized influence of
rank-1 results and rewards consistent mid-rank appearances.

This module is pure rank math — zero dependency on embeddings or the
vector store.  It accepts any number of ranked lists so it composes
cleanly with additional retrieval strategies added in the future.
"""

from __future__ import annotations

from collections import defaultdict

from src.vector_store import ScoredChunk

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = RRF_K,
    top_k: int = 3,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists into a single ranked list using RRF.

    Parameters
    ----------
    ranked_lists : list of ranked ScoredChunk lists (one per strategy)
    k            : RRF dampening constant (default 60)
    top_k        : how many results to return

    Returns
    -------
    Merged list of ScoredChunks sorted by descending RRF score.
    The .score field contains the RRF score (not cosine similarity).
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_lookup: dict[str, ScoredChunk] = {}

    for ranked in ranked_lists:
        for chunk in ranked:
            rrf_scores[chunk.chunk_id] += 1.0 / (k + chunk.rank)
            chunk_lookup[chunk.chunk_id] = chunk

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (chunk_id, score) in enumerate(merged[:top_k], start=1):
        original = chunk_lookup[chunk_id]
        results.append(
            ScoredChunk(
                chunk_id=chunk_id,
                text=original.text,
                score=round(score, 6),
                rank=rank,
            )
        )
    return results