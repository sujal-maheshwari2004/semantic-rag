"""
Tests for GenerativeModel (HyDE-based query expander).
Verifies the mock contract matches the Vertex AI SDK signature.

Note: test_expansion_includes_original_query and test_stopwords_not_in_expansion
have been updated because the HyDE mock returns a *hypothetical answer passage*
rather than appending the original query as a prefix. The production swap
(real GenerativeModel) would also return a passage, not an augmented query,
so these tests now reflect the correct contract.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.query_expander import GenerativeModel


@pytest.fixture
def model():
    return GenerativeModel("gemini-1.5-pro")


def test_generate_content_returns_response_with_text(model):
    response = model.generate_content("How does the system handle peak load?")
    assert hasattr(response, "text")
    assert isinstance(response.text, str)


def test_expansion_is_non_empty_string(model):
    """HyDE returns a passage, not a prefix-augmented query."""
    query = "How does the system handle peak load?"
    response = model.generate_content(query)
    assert len(response.text) > 0


def test_expansion_is_longer_than_query(model):
    """The hypothesis passage should be substantially longer than the raw query."""
    query = "How does the system handle peak load?"
    response = model.generate_content(query)
    assert len(response.text) > len(query)


def test_expansion_is_deterministic(model):
    query = "What mechanisms ensure data consistency across nodes?"
    r1 = model.generate_content(query)
    r2 = model.generate_content(query)
    assert r1.text == r2.text


def test_known_query_returns_domain_relevant_passage(model):
    """HyDE passages should contain domain vocabulary, not WordNet noise."""
    response = model.generate_content("How does the system handle peak load?")
    text = response.text.lower()
    # at least one load-handling term must appear in the hypothesis
    assert any(term in text for term in ["autoscal", "queue", "circuit", "cache", "replica"])


def test_data_consistency_query_returns_consensus_terms(model):
    response = model.generate_content("What mechanisms ensure data consistency across nodes?")
    text = response.text.lower()
    assert any(term in text for term in ["raft", "paxos", "quorum", "consensus", "replica"])


def test_inference_query_returns_serving_terms(model):
    response = model.generate_content("How is model inference optimised at serving time?")
    text = response.text.lower()
    assert any(term in text for term in ["quantis", "batch", "gpu", "kv cache", "speculative"])


def test_unknown_query_returns_default_hypothesis(model):
    """Queries that match no template still return a non-empty fallback."""
    response = model.generate_content("What is the boiling point of nitrogen?")
    assert isinstance(response.text, str)
    assert len(response.text) > 10


def test_model_name_stored(model):
    assert model.model_name == "gemini-1.5-pro"


def test_mock_patch_replaces_generate_content():
    """Verify pytest-mock can patch generate_content — production swap pattern."""
    with patch.object(GenerativeModel, "generate_content") as mock_gc:
        mock_gc.return_value = MagicMock(text="mocked hypothesis passage")
        m = GenerativeModel()
        result = m.generate_content("any query")
        assert result.text == "mocked hypothesis passage"
        mock_gc.assert_called_once_with("any query")