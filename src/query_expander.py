"""
Query expander — mocks vertexai.generative_models.GenerativeModel.

Upgraded from naive WordNet synonym appending to a HyDE
(Hypothetical Document Embedding) approach: the mock generates a
short, plausible technical passage that *answers* the query, exactly
as a real GenerativeModel would when prompted with
"Write a technical paragraph that answers: {query}".

Why HyDE is better than synonym expansion:
    - The hypothesis occupies the same embedding space as corpus
      passages, closing the vocabulary gap between a short question
      and long technical text.
    - Domain register is preserved: generated text uses the same
      jargon as the corpus instead of WordNet's general-language
      synonyms ("burden", "prison term", "efflorescence").
    - A real GenerativeModel swap requires zero logic changes — just
      swap the import and the mock's _generate() body becomes the
      live API call.

Production swap:
    Replace GenerativeModel with the real vertexai import:
        from vertexai.generative_models import GenerativeModel
    The rest of the pipeline is unchanged.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HyDE templates — stand-in for what a live Gemini call would return.
# Keyed by normalised query substring for deterministic offline testing.
# A real model would generate these dynamically.
# ---------------------------------------------------------------------------

_HYDE_TEMPLATES: dict[str, str] = {
    "peak load": (
        "Systems handle peak load through horizontal autoscaling that provisions "
        "additional service replicas when CPU utilisation or queue depth exceeds a "
        "threshold. Message queues absorb burst traffic and apply backpressure to "
        "upstream producers. Circuit breakers prevent overloaded dependencies from "
        "cascading failures. Multi-tier caching reduces database pressure during "
        "traffic spikes by serving hot keys from in-process or Redis caches."
    ),
    "data consistency": (
        "Data consistency across distributed nodes is maintained through consensus "
        "protocols such as Raft and Paxos, which commit writes only after a quorum "
        "of replicas acknowledge the entry. Read-your-writes consistency is enforced "
        "by routing reads to the leader or tagging requests with a logical timestamp. "
        "Cache invalidation is coordinated via pub-sub events to avoid stale reads "
        "from multi-tier caches."
    ),
    "model inference": (
        "Model inference at serving time is optimised through quantisation, which "
        "reduces weight precision from FP32 to INT8 with minimal accuracy loss, "
        "operator fusion that collapses consecutive GPU kernels, and dynamic batching "
        "that groups concurrent requests to maximise hardware utilisation. Speculative "
        "decoding uses a small draft model to propose tokens verified by the main model "
        "in parallel. Paged attention manages KV cache memory for thousands of concurrent "
        "sequences without fragmentation."
    ),
}

_DEFAULT_HYPOTHESIS = (
    "This system component is designed for high performance and reliability in "
    "distributed cloud environments, leveraging modern techniques for scalability, "
    "fault tolerance, and low-latency operation."
)


class _GenerativeResponse:
    """Mirrors the response object returned by GenerativeModel.generate_content()."""

    def __init__(self, text: str) -> None:
        self.text = text


class GenerativeModel:
    """
    Mocks vertexai.generative_models.GenerativeModel using HyDE.

    generate_content() accepts a query string and returns a response
    whose .text attribute is a short hypothetical document that answers
    the query — the same output a real Gemini call with a HyDE prompt
    would produce.  Expansion is deterministic and fully offline.
    """

    def __init__(self, model_name: str = "gemini-1.5-pro") -> None:
        self.model_name = model_name

    def generate_content(self, query: str) -> _GenerativeResponse:
        hypothesis = self._generate(query)
        return _GenerativeResponse(text=hypothesis)

    # ------------------------------------------------------------------
    # internal generation logic
    # ------------------------------------------------------------------

    def _generate(self, query: str) -> str:
        """
        Return a hypothetical answer passage for the query.

        In production this becomes:
            prompt = f"Write a technical paragraph that answers: {query}"
            return self._model.generate_content(prompt).text
        """
        lower = query.lower()
        for keyword, template in _HYDE_TEMPLATES.items():
            if keyword in lower:
                return template
        return _DEFAULT_HYPOTHESIS