"""
Evaluator — computes IR metrics against a ground-truth relevance map.

Metrics
-------
MRR  (Mean Reciprocal Rank):
    1 / rank_of_first_relevant_result
    0.0 if no relevant result appears in top_k.

Precision@K:
    |relevant ∩ retrieved| / K

Ground truth file format (data/ground_truth.json):
    { "query string": ["chunk_id_1", "chunk_id_2", ...], ... }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.vector_store import ScoredChunk


@dataclass
class EvalResult:
    query: str
    strategy: str
    mrr: float
    precision_at_k: float
    k: int


class Evaluator:
    def __init__(self, ground_truth_path: str = "data/ground_truth.json") -> None:
        raw = json.loads(Path(ground_truth_path).read_text())
        self._ground_truth: dict[str, set[str]] = {
            q: set(ids) for q, ids in raw.items()
        }

    def score(
        self,
        query: str,
        strategy: str,
        chunks: list[ScoredChunk],
    ) -> EvalResult:
        relevant = self._ground_truth.get(query, set())
        k = len(chunks)

        # MRR
        mrr = 0.0
        for chunk in chunks:
            if chunk.chunk_id in relevant:
                mrr = 1.0 / chunk.rank
                break

        # Precision@K
        retrieved_ids = {c.chunk_id for c in chunks}
        hits = len(relevant & retrieved_ids)
        precision_at_k = hits / k if k > 0 else 0.0

        return EvalResult(
            query=query,
            strategy=strategy,
            mrr=round(mrr, 4),
            precision_at_k=round(precision_at_k, 4),
            k=k,
        )