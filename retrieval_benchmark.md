# Retrieval Benchmark — Strategy A vs B vs C

## Setup

| | |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (local, simulates `textembedding-gecko@003`) |
| Vector store | FAISS `IndexFlatIP` (cosine similarity via L2-normalised vectors) |
| Strategy A | Raw vector search — query embedded as-is |
| Strategy B | Synonym expansion via mocked `GenerativeModel` (WordNet nouns) |
| Strategy C | Reciprocal Rank Fusion of A and B (k=60) |
| Corpus | 10 technical paragraphs across distributed systems and ML serving |
| Queries | 3 complex domain queries |
| Metrics | MRR (Mean Reciprocal Rank), Precision@3 |

## Query: _How does the system handle peak load?_

**Strategy B expanded query:** `How does the system handle peak load? acme apex arrangement bill bloom blossom burden cargo consignment crest [...]`

| Strategy | Rank 1 | Rank 2 | Rank 3 | MRR | P@3 |
|---|---|---|---|---|---|
| A — Raw vector search | chunk_00 | chunk_02 | chunk_06 | 1.0 | 0.6667 |
| B — Synonym expansion | chunk_00 | chunk_06 | chunk_02 | 1.0 | 0.6667 |
| C — RRF (A ∪ B) | chunk_00 | chunk_02 | chunk_06 | 1.0 | 0.6667 |

## Query: _What mechanisms ensure data consistency across nodes?_

**Strategy B expanded query:** `What mechanisms ensure data consistency across nodes? body client consistence data point datum eubstance guest [...]`

| Strategy | Rank 1 | Rank 2 | Rank 3 | MRR | P@3 |
|---|---|---|---|---|---|
| A — Raw vector search | chunk_05 | chunk_06 | chunk_09 | 1.0 | 0.6667 |
| B — Synonym expansion | chunk_05 | chunk_06 | chunk_07 | 1.0 | 1.0 |
| C — RRF (A ∪ B) | chunk_05 | chunk_06 | chunk_09 | 1.0 | 0.6667 |

## Query: _How is model inference optimised at serving time?_

**Strategy B expanded query:** `How is model inference optimised at serving time? clip clock time fourth dimension illation meter metre prison [...]`

| Strategy | Rank 1 | Rank 2 | Rank 3 | MRR | P@3 |
|---|---|---|---|---|---|
| A — Raw vector search | chunk_04 | chunk_00 | chunk_06 | 1.0 | 0.3333 |
| B — Synonym expansion | chunk_04 | chunk_07 | chunk_00 | 1.0 | 0.3333 |
| C — RRF (A ∪ B) | chunk_04 | chunk_00 | chunk_07 | 1.0 | 0.3333 |

## Similarity Metric: Cosine vs Euclidean

Cosine similarity is preferred for sentence embeddings because vectors vary
in magnitude with sequence length. L2-normalising all vectors and using inner
product (FAISS `IndexFlatIP`) is mathematically equivalent to cosine similarity
while remaining compatible with Vertex AI Matching Engine's default metric.
Euclidean distance would penalise shorter queries against longer corpus chunks,
introducing a length bias unrelated to semantic content.

## Why Strategy B Sometimes Underperforms

WordNet synonym expansion is domain-agnostic. For highly specific technical
queries, added synonyms can drift from the corpus vocabulary
(e.g. 'load' -> 'burden', 'cargo' instead of 'traffic', 'throughput').
A real `GenerativeModel` would expand with domain awareness, producing
hypothesis text that stays within the technical register of the corpus.
This is the primary motivation for using RRF (Strategy C): it hedges against
Strategy B's occasional drift by requiring agreement across both ranked lists.

## Vertex AI Migration Path

| Component | Local | Vertex AI |
|---|---|---|
| Embedder | `sentence-transformers` | `TextEmbeddingModel.from_pretrained('textembedding-gecko@003')` |
| Vector store | `faiss.IndexFlatIP` | `MatchingEngineIndexEndpoint.find_neighbors()` |
| Query expansion | Mocked `GenerativeModel` (WordNet) | `GenerativeModel('gemini-1.5-pro')` |
| Auth | None | Workload Identity Federation (no service account keys) |
| Index type | Exact (brute force) | `tree-ah` for >1M vectors, `brute_force` for dev |

The swap requires changing two import lines and one constructor argument.
All retrieval logic, fusion, evaluation, and benchmark code is unchanged.