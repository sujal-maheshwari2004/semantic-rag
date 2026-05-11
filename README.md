# semantic-rag

A local Retrieval-Augmented Generation (RAG) pipeline demonstrating semantic search with three retrieval strategies, evaluated against ground-truth relevance labels.

Built as a GCP Gen AI engineering assessment. All GCP SDK calls (`TextEmbeddingModel`, `GenerativeModel`) are mocked locally using `sentence-transformers` and WordNet — the production swap is a one-line import change.

---

## Architecture

```
data/corpus.py          10 technical paragraphs (the knowledge base)
data/ground_truth.json  Relevance labels for 3 benchmark queries

src/embedder.py         Mocks vertexai TextEmbeddingModel → sentence-transformers locally
src/cache.py            sha256 embedding cache (disk-backed, avoids re-embedding)
src/vector_store.py     FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
src/query_expander.py   Mocks vertexai GenerativeModel → WordNet noun synonym expansion
src/fusion.py           Reciprocal Rank Fusion (RRF, k=60)
src/retriever.py        Strategy A (raw), B (synonym-expanded), C (RRF)
src/pipeline.py         Orchestrator — ingest → embed → store → retrieve
src/evaluator.py        MRR and Precision@K against ground truth

benchmark/run.py        Runs 3 queries × 3 strategies, outputs table + JSON + markdown
```

---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| A — Raw vector search | Query embedded as-is, searched against FAISS index |
| B — Synonym expansion | Query nouns expanded with WordNet synonyms via mocked `GenerativeModel`, then embedded |
| C — RRF (A ∪ B) | Runs A and B independently, merges ranked lists using Reciprocal Rank Fusion |

**Why RRF?** Strategy B's WordNet expansion is domain-agnostic and can drift (e.g. `load → burden, cargo` instead of `traffic, throughput`). RRF hedges against this by rewarding chunks that appear highly in *both* ranked lists, requiring cross-strategy agreement.

---

## Similarity Metric

**Cosine similarity** via FAISS `IndexFlatIP` on L2-normalised vectors.

Sentence embedding magnitudes vary with sequence length — a short query would score artificially low against long corpus chunks under Euclidean distance. L2 normalisation removes this length bias, making inner product equivalent to cosine similarity. This also matches Vertex AI Matching Engine's default metric, so the production swap requires no metric reconfiguration.

---

## Setup

```bash
# requires uv (https://docs.astral.sh/uv)
git clone <repo>
cd semantic-rag

uv sync

# download WordNet for query expansion (one-time)
uv run python -c "import nltk; nltk.download('wordnet')"
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

44 tests across 5 modules. The `conftest.py` patches `SentenceTransformer` at session scope — tests are fully offline and hermetic.

---

## Running the Benchmark

```bash
uv run python -m benchmark.run
```

Outputs:
- Rich table to stdout (3 queries × 3 strategies, MRR + P@3)
- `benchmark/results.json` — machine-readable full results
- `retrieval_benchmark.md` — committed benchmark report

---

## Vertex AI Migration Path

| Component | Local (this repo) | Production (Vertex AI) |
|---|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` | `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
| Vector store | `faiss.IndexFlatIP` | `MatchingEngineIndexEndpoint.find_neighbors()` |
| Query expansion | Mocked `GenerativeModel` (WordNet) | `GenerativeModel("gemini-1.5-pro")` |
| Auth | None | Workload Identity Federation |
| Index type | Exact / brute force | `tree-ah` for >1M vectors, `brute_force` for dev |

The swap requires changing **two import lines** and **one constructor argument**. All retrieval logic, RRF fusion, evaluation, and benchmark code is unchanged.

For Matching Engine specifically:
- Build index offline from a Cloud Storage corpus
- Deploy to a dedicated `IndexEndpoint`
- Replace `store.search(vec, top_k)` with `index_endpoint.find_neighbors(queries=[vec], num_neighbors=top_k)`
- Auth via Workload Identity Federation — no service account key files

---

## Project Structure

```
semantic-rag/
├── src/
│   ├── __init__.py
│   ├── cache.py
│   ├── embedder.py
│   ├── evaluator.py
│   ├── fusion.py
│   ├── pipeline.py
│   ├── query_expander.py
│   ├── retriever.py
│   └── vector_store.py
├── data/
│   ├── corpus.py
│   └── ground_truth.json
├── tests/
│   ├── conftest.py
│   ├── test_embedder.py
│   ├── test_fusion.py
│   ├── test_pipeline.py
│   ├── test_query_expander.py
│   ├── test_retriever.py
│   └── test_vector_store.py
├── benchmark/
│   ├── run.py
│   └── results.json
├── retrieval_benchmark.md
├── pyproject.toml
└── README.md
```