"""
Tests for GenerativeModel (mocked query expander).
Verifies the mock contract matches the Vertex AI SDK signature.
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


def test_expansion_includes_original_query(model):
    query = "How does the system handle peak load?"
    response = model.generate_content(query)
    assert query in response.text


def test_expansion_adds_synonyms(model):
    query = "How does the system handle peak load?"
    response = model.generate_content(query)
    assert len(response.text) > len(query)


def test_expansion_is_deterministic(model):
    query = "What mechanisms ensure data consistency across nodes?"
    r1 = model.generate_content(query)
    r2 = model.generate_content(query)
    assert r1.text == r2.text


def test_model_name_stored(model):
    assert model.model_name == "gemini-1.5-pro"


def test_mock_patch_replaces_generate_content():
    """Verify pytest-mock can patch generate_content — production swap pattern."""
    with patch.object(GenerativeModel, "generate_content") as mock_gc:
        mock_gc.return_value = MagicMock(text="mocked expansion")
        m = GenerativeModel()
        result = m.generate_content("any query")
        assert result.text == "mocked expansion"
        mock_gc.assert_called_once_with("any query")


def test_stopwords_not_in_expansion(model):
    response = model.generate_content("How does the system handle peak load?")
    # common stopwords should not appear as standalone added terms
    expansion_only = response.text.replace("How does the system handle peak load?", "").lower()
    for stopword in ["the", "does", "how"]:
        assert stopword not in expansion_only.split()