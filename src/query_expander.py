"""
Query expander — mocks vertexai.generative_models.GenerativeModel.

Instead of a hardcoded stub, the mock does something algorithmically
meaningful: it expands the query with WordNet synonyms for content
words (nouns only), simulating the semantic broadening a real LLM
would perform via HyDE or query rewriting.

Why nouns only:
    Nouns carry the most domain-specific signal in technical queries.
    Expanding verbs ('handle' → 'wield', 'grip') introduces noise.

Production swap:
    Replace GenerativeModel with the real vertexai import:
        from vertexai.generative_models import GenerativeModel
    The rest of the pipeline is unchanged.
"""

from __future__ import annotations

import nltk
from nltk.corpus import stopwords, wordnet

nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)

STOP = set(stopwords.words("english"))
MAX_SYNONYMS = 12


class _GenerativeResponse:
    """Mirrors the response object returned by GenerativeModel.generate_content()."""

    def __init__(self, text: str) -> None:
        self.text = text


class GenerativeModel:
    """
    Mocks vertexai.generative_models.GenerativeModel.

    generate_content() accepts a prompt string and returns a response
    whose .text attribute contains the expanded query.  The expansion
    uses WordNet synonym lookup — deterministic, offline, no API calls.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        self.model_name = model_name

    def generate_content(self, prompt: str) -> _GenerativeResponse:
        expanded = self._expand(prompt)
        return _GenerativeResponse(text=expanded)

    # ------------------------------------------------------------------
    # internal expansion logic
    # ------------------------------------------------------------------

    def _expand(self, query: str) -> str:
        tokens = nltk.word_tokenize(query)
        tagged = nltk.pos_tag(tokens)

        synonyms: set[str] = set()
        for word, pos in tagged:
            if word.lower() in STOP or not word.isalpha():
                continue
            if not pos.startswith("NN"):   # nouns only
                continue
            for syn in wordnet.synsets(word, pos=wordnet.NOUN):
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ").lower()
                    if candidate != word.lower() and candidate not in STOP:
                        synonyms.add(candidate)

        capped = sorted(synonyms)[:MAX_SYNONYMS]
        expansion = " ".join(capped)
        return f"{query} {expansion}".strip()