"""Tests for TextEmbeddingModel."""

import numpy as np
import pytest

from src.embedder import TextEmbedding, TextEmbeddingModel


@pytest.fixture
def model(tmp_path):
    return TextEmbeddingModel(cache_dir=str(tmp_path))


def test_get_embeddings_returns_correct_count(model):
    texts = ["hello world", "semantic search", "vector embeddings"]
    results = model.get_embeddings(texts)
    assert len(results) == 3


def test_get_embeddings_returns_text_embedding_instances(model):
    results = model.get_embeddings(["test"])
    assert isinstance(results[0], TextEmbedding)
    assert isinstance(results[0].values, list)


def test_embedding_dimension(model):
    results = model.get_embeddings(["test sentence"])
    assert len(results[0].values) == 384


def test_vectors_are_l2_normalised(model):
    vec = model.embed_one("normalisation check")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5


def test_embed_many_shape(model):
    texts = ["a", "b", "c"]
    matrix = model.embed_many(texts)
    assert matrix.shape == (3, 384)


def test_cache_hit_returns_same_vector(model):
    text = "cache consistency test"
    v1 = model.embed_one(text)
    v2 = model.embed_one(text)
    np.testing.assert_array_almost_equal(v1, v2)


def test_from_pretrained_returns_model():
    m = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    assert isinstance(m, TextEmbeddingModel)