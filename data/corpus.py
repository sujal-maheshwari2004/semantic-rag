"""
Technical corpus these are 10 paragraphs covering distributed systems,
ML serving, and cloud infrastructure. Each entry is one chunk and related
to a specific technical concept.
"""

CHUNKS = [
    {
        "id": "chunk_00",
        "topic": "autoscaling",
        "text": (
            "Horizontal autoscaling dynamically adjusts the number of running service "
            "instances based on observed CPU utilisation, memory pressure, or custom "
            "metrics such as queue depth. During peak load, the orchestrator provisions "
            "additional replicas within seconds, distributing incoming traffic across "
            "the expanded pool. Scale-in policies include cooldown windows to prevent "
            "thrashing when load oscillates around a threshold."
        ),
    },
    {
        "id": "chunk_01",
        "topic": "circuit_breaker",
        "text": (
            "The circuit breaker pattern protects downstream services from cascading "
            "failures. When the error rate for a dependency exceeds a configurable "
            "threshold the breaker trips to the open state, returning fallback responses "
            "immediately without attempting the remote call. After a half-open probe "
            "succeeds the breaker resets, restoring normal traffic. This mechanism "
            "is essential for maintaining system stability under partial outages."
        ),
    },
    {
        "id": "chunk_02",
        "topic": "queue_backpressure",
        "text": (
            "Message queues decouple producers from consumers and absorb burst traffic "
            "that would otherwise overwhelm synchronous endpoints. Backpressure signals "
            "propagate upstream when consumer lag exceeds a high-water mark, allowing "
            "producers to shed load gracefully. Dead-letter queues capture messages that "
            "exceed the maximum retry count, enabling offline inspection and replay "
            "without data loss during peak demand periods."
        ),
    },
    {
        "id": "chunk_03",
        "topic": "vector_embeddings",
        "text": (
            "Dense vector embeddings represent semantic meaning as points in a "
            "high-dimensional space. Models such as sentence-transformers encode "
            "sentences into fixed-length vectors where cosine similarity correlates "
            "with semantic relatedness. These representations power retrieval-augmented "
            "generation pipelines, recommendation systems, and duplicate detection by "
            "enabling sub-linear approximate nearest-neighbour search over millions "
            "of documents."
        ),
    },
    {
        "id": "chunk_04",
        "topic": "model_serving",
        "text": (
            "Optimising model inference at serving time involves several complementary "
            "techniques: quantisation reduces weight precision from FP32 to INT8 with "
            "minimal accuracy loss, operator fusion collapses consecutive kernels into "
            "a single GPU launch, and dynamic batching groups concurrent requests to "
            "maximise hardware utilisation. Triton Inference Server and TorchServe "
            "expose these optimisations behind a unified gRPC and HTTP interface."
        ),
    },
    {
        "id": "chunk_05",
        "topic": "data_consistency",
        "text": (
            "Distributed databases achieve consistency through consensus protocols such "
            "as Raft and Paxos. A write is committed only after a quorum of replicas "
            "acknowledge the entry, guaranteeing that any subsequent read from a "
            "majority quorum reflects the latest value. Read-your-writes consistency "
            "can be enforced by routing reads to the leader replica or by tagging "
            "requests with a logical timestamp that followers must have applied."
        ),
    },
    {
        "id": "chunk_06",
        "topic": "caching",
        "text": (
            "Multi-tier caching reduces latency and database load. An in-process LRU "
            "cache serves hot keys with nanosecond latency, a shared Redis cluster "
            "provides microsecond access across service instances, and a CDN edge cache "
            "absorbs read traffic for public assets. Cache invalidation is coordinated "
            "through pub-sub events emitted on write, ensuring consistency without "
            "requiring a distributed lock on every read path."
        ),
    },
    {
        "id": "chunk_07",
        "topic": "observability",
        "text": (
            "Effective observability combines structured logs, metrics, and distributed "
            "traces. Each request carries a propagated trace context that links spans "
            "across service boundaries, enabling latency attribution to individual "
            "components. Anomaly detection on golden signals — request rate, error rate, "
            "latency, and saturation — triggers automated runbooks before user-facing "
            "impact is detected, reducing mean time to recovery."
        ),
    },
    {
        "id": "chunk_08",
        "topic": "faiss_index",
        "text": (
            "FAISS provides efficient similarity search over dense vectors. The "
            "IndexFlatIP index performs exact inner-product search, equivalent to "
            "cosine similarity on L2-normalised vectors, and serves as a correctness "
            "baseline. For production scale, IndexIVFPQ partitions the vector space "
            "into Voronoi cells and applies product quantisation within each cell, "
            "achieving sub-linear query time at the cost of a small approximation error "
            "controllable via the nprobe parameter."
        ),
    },
    {
        "id": "chunk_09",
        "topic": "vertex_ai",
        "text": (
            "Vertex AI Matching Engine provides a managed approximate nearest-neighbour "
            "service backed by Google's ScaNN algorithm. Indexes are built offline from "
            "a Cloud Storage corpus and deployed to a dedicated IndexEndpoint. At query "
            "time, find_neighbors() accepts a batch of query embeddings and returns "
            "ranked neighbours with sub-millisecond p99 latency at billion-vector scale. "
            "Incremental updates stream new vectors without requiring a full index rebuild."
        ),
    },
]

CHUNK_IDS = [c["id"] for c in CHUNKS]
TEXTS = [c["text"] for c in CHUNKS]