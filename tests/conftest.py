"""
conftest.py — shared pytest fixtures.

SentenceTransformer is mocked at session scope so that tests never
attempt to reach HuggingFace Hub.  The mock returns deterministic
random-but-L2-normalised 384-dim vectors, which is sufficient for
testing all retrieval and evaluation logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _make_mock_st() -> MagicMock:
    """Return a MagicMock that behaves like SentenceTransformer."""
    rng = np.random.RandomState(42)

    def fake_encode(texts, normalize_embeddings=True, **kwargs):
        vecs = rng.randn(len(texts), 384).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= norms
        return vecs

    mock = MagicMock()
    mock.encode.side_effect = fake_encode
    return mock


@pytest.fixture(autouse=True, scope="session")
def mock_sentence_transformer():
    """Patch SentenceTransformer for the entire test session."""
    mock_st_instance = _make_mock_st()
    with patch(
        "src.embedder.SentenceTransformer",
        return_value=mock_st_instance,
    ):
        yield mock_st_instance