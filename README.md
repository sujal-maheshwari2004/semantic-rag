# semantic-rag

A local Retrieval-Augmented Generation (RAG) pipeline demonstrating semantic search with three retrieval strategies, evaluated against ground-truth relevance labels.

Built as a GCP Gen AI engineering assessment. All GCP SDK calls (`TextEmbeddingModel`, `GenerativeModel`) are mocked locally using `sentence-transformers` and a HyDE-style template expander — the production swap is a one-line import change.

---

## Architecture

```
data/corpus.py          100 technical paragraphs (the knowledge base)
data/ground_truth.json  Relevance labels for 3 benchmark queries

src/embedder.py         Mocks vertexai TextEmbeddingModel → sentence-transformers locally
src/cache.py            sha256 embedding cache (disk-backed, thread-safe, atomic writes)
src/vector_store.py     FAISS IndexFlatIP (cosine similarity via L2-normalised vectors)
src/query_expander.py   Mocks vertexai GenerativeModel → HyDE hypothetical document generation
src/fusion.py           Reciprocal Rank Fusion (RRF, k=60)
src/retriever.py        Strategy A (raw), B (HyDE-expanded), C (RRF)
src/pipeline.py         Orchestrator — ingest → embed → store → retrieve (sync + async)
src/evaluator.py        MRR and Precision@K against ground truth

benchmark/run.py        Runs 3 queries × 3 strategies, outputs table + JSON + markdown
```

---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| A — Raw vector search | Query embedded as-is, searched against FAISS index |
| B — HyDE expansion | A hypothetical answer passage is generated via mocked `GenerativeModel`, then embedded |
| C — RRF (A ∪ B) | Runs A and B independently, merges ranked lists using Reciprocal Rank Fusion |

**Why HyDE over WordNet?** WordNet expands queries with general-language synonyms that drift from technical vocabulary (`load → burden, cargo` instead of `traffic, throughput`). HyDE generates a short passage in the same register as the corpus, closing the vocabulary gap between a terse question and a long technical chunk.

**Why RRF?** Strategy B's HyDE hypothesis is a static template in this mock; a live model introduces stochastic variation. RRF hedges against any single-strategy drift by rewarding chunks that appear highly in *both* ranked lists.

---

## Design Decisions

### Similarity Metric

**Cosine similarity** via FAISS `IndexFlatIP` on L2-normalised vectors.

Sentence embedding magnitudes vary with sequence length — a short query scores artificially low against long corpus chunks under Euclidean distance. L2 normalisation removes this length bias, making inner product equivalent to cosine similarity. This also matches Vertex AI Matching Engine's default metric.

### Corpus Size

100 chunks across distributed systems, ML serving, cloud infrastructure, databases, networking, and security. At 10 chunks MRR is trivially 1.0 for every strategy — the 100-chunk corpus introduces enough distractor content that strategy differences in Precision@K become visible and meaningful.

### Async Support

`RAGPipeline` exposes `async_ingest()` and `async_query()` coroutines that offload CPU-bound work (embedding + FAISS search) to a shared `ThreadPoolExecutor`, keeping the event loop free. This is the correct pattern for FastAPI / aiohttp serving where multiple concurrent requests share a single loop.

```python
pipeline = RAGPipeline()
await pipeline.async_ingest(CHUNKS)
result = await pipeline.async_query("peak load", strategy="C", top_k=5)
```

### Thread-Safe Cache

`EmbeddingCache` uses a `threading.Lock` to guard all in-memory mutations and writes disk files via `os.replace()` (atomic rename on POSIX and Windows). A crash mid-write never leaves a corrupt `.npy` or `.json` file.

### Strategy Dispatch

Pipeline dispatch is driven by `_STRATEGY_REGISTRY`, a dict mapping strategy keys to `(class, needs_expander)` tuples. There are no `if/elif` branches — adding a new strategy requires only a single registry entry.

---

## Setup

```bash
# requires uv (https://docs.astral.sh/uv)
git clone <repo>
cd semantic-rag
uv sync
```

No NLTK downloads required — the HyDE expander is pure Python with no external corpora.

---

## Running Tests

```bash
uv run pytest tests/ -v
```

Tests cover: embedder, cache (including thread-safety), fusion, retriever, pipeline (sync + async), vector store, and query expander. All tests are fully offline and hermetic.

---

## Running the Benchmark

```bash
uv run python -m benchmark.run
```

Outputs:
- Rich table to stdout (3 queries × 3 strategies, MRR + P@K)
- `benchmark/results.json` — machine-readable full results
- `retrieval_benchmark.md` — committed benchmark report

---

## Vertex AI Migration Path

| Component | Local (this repo) | Production (Vertex AI) |
|---|---|---|
| Embedder | `sentence-transformers/all-MiniLM-L6-v2` | `TextEmbeddingModel.from_pretrained("textembedding-gecko@003")` |
| Vector store | `faiss.IndexFlatIP` | `MatchingEngineIndexEndpoint.find_neighbors()` |
| Query expansion | HyDE template mock | `GenerativeModel("gemini-1.5-pro")` with HyDE prompt |
| Cache | Disk-backed `.npy` + `.json` | Cloud Memorystore (Redis) |
| Auth | None | Workload Identity Federation |
| Index type | Exact / brute force | `tree-ah` for >1M vectors, `brute_force` for dev |
| Async client | `ThreadPoolExecutor` | Native async google-cloud-aiplatform stubs |

The swap requires changing **two import lines** and **one constructor argument**. All retrieval logic, RRF fusion, evaluation, and benchmark code is unchanged.

For Matching Engine specifically:
- Build index offline from a Cloud Storage corpus
- Deploy to a dedicated `IndexEndpoint`
- Replace `store.search(vec, top_k)` with `index_endpoint.find_neighbors(queries=[vec], num_neighbors=top_k)`
- Auth via Workload Identity Federation — no service account key files