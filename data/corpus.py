"""
Technical corpus — 100 paragraphs covering distributed systems,
ML serving, cloud infrastructure, databases, networking, and security.

The first 10 chunks are identical to the original so existing ground-truth
labels remain valid.  Chunks 10-99 add breadth and distractor content that
makes MRR < 1.0 achievable and strategy differentiation meaningful.
"""

CHUNKS = [
    # ── original 10 ──────────────────────────────────────────────────────────
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
    # ── distractor / enrichment chunks 10-99 ─────────────────────────────────
    {
        "id": "chunk_10",
        "topic": "load_balancing",
        "text": (
            "Layer-7 load balancers inspect HTTP headers and URL paths to route requests "
            "to the appropriate backend pool. Least-connection and weighted round-robin "
            "algorithms distribute traffic according to real-time server capacity. "
            "Health checks remove unhealthy instances from rotation within seconds, "
            "preventing user-facing errors during rolling deployments or sudden failures."
        ),
    },
    {
        "id": "chunk_11",
        "topic": "service_mesh",
        "text": (
            "A service mesh such as Istio or Linkerd injects a sidecar proxy alongside "
            "every workload container. The data plane intercepts all inbound and outbound "
            "traffic, enforcing mutual TLS, applying rate limits, and emitting telemetry "
            "without requiring application code changes. The control plane distributes "
            "routing rules and certificate rotations across the fleet."
        ),
    },
    {
        "id": "chunk_12",
        "topic": "kubernetes_scheduling",
        "text": (
            "The Kubernetes scheduler assigns pods to nodes by evaluating resource "
            "requests against available capacity, respecting taints, tolerations, and "
            "affinity rules. Bin-packing heuristics maximise node utilisation while "
            "topology-spread constraints distribute replicas across failure domains. "
            "Priority classes ensure critical workloads preempt lower-priority pods "
            "under resource pressure."
        ),
    },
    {
        "id": "chunk_13",
        "topic": "grpc_streaming",
        "text": (
            "gRPC bidirectional streaming enables low-latency, multiplexed communication "
            "over a single HTTP/2 connection. Clients and servers exchange messages "
            "independently without waiting for a full request-response cycle. Flow "
            "control at the HTTP/2 layer prevents fast producers from overwhelming slow "
            "consumers, and protobuf serialisation keeps payload size small compared "
            "to JSON equivalents."
        ),
    },
    {
        "id": "chunk_14",
        "topic": "event_sourcing",
        "text": (
            "Event sourcing persists state as an immutable, append-only sequence of "
            "domain events rather than mutable rows. The current state of any aggregate "
            "is derived by replaying its event stream from the beginning or from a "
            "snapshot. This audit log enables temporal queries, retroactive bug fixes "
            "via event replay, and decoupled downstream projections that consume "
            "the same stream asynchronously."
        ),
    },
    {
        "id": "chunk_15",
        "topic": "saga_pattern",
        "text": (
            "The Saga pattern coordinates long-running distributed transactions without "
            "a two-phase commit. Each step publishes a domain event on success; a "
            "compensating transaction rolls back completed steps if a later step fails. "
            "Choreography-based sagas rely on event routing alone, while orchestration-"
            "based sagas use a central coordinator that explicitly invokes each "
            "participant and tracks global state."
        ),
    },
    {
        "id": "chunk_16",
        "topic": "blue_green_deployment",
        "text": (
            "Blue-green deployment maintains two identical production environments. "
            "The new version is deployed to the idle environment and smoke-tested "
            "before a router switch transfers all traffic instantaneously. Rollback "
            "requires only a single routing change, reducing mean time to recovery "
            "to seconds. The primary cost is the double infrastructure footprint "
            "during the transition window."
        ),
    },
    {
        "id": "chunk_17",
        "topic": "canary_release",
        "text": (
            "Canary releases route a small percentage of production traffic to a new "
            "version while the majority of users remain on the stable release. "
            "Automated rollout gates compare error rate, latency p99, and business "
            "metrics between the canary and baseline cohorts. If the canary degrades "
            "any metric beyond a threshold, the pipeline halts and rolls back "
            "automatically without manual intervention."
        ),
    },
    {
        "id": "chunk_18",
        "topic": "feature_flags",
        "text": (
            "Feature flags decouple code deployment from feature activation. A "
            "centralised flag evaluation service resolves flag state per request based "
            "on user attributes, percentage rollout, or explicit allow-lists. Flags "
            "enable dark launches, A/B experiments, and emergency kill-switches that "
            "disable a misbehaving code path without a new deployment."
        ),
    },
    {
        "id": "chunk_19",
        "topic": "rate_limiting",
        "text": (
            "Token-bucket and sliding-window rate limiters enforce per-client or "
            "per-endpoint request quotas. Distributed implementations store token "
            "counts in Redis using atomic Lua scripts to prevent race conditions. "
            "Adaptive rate limiting backs off automatically when upstream latency "
            "increases, protecting dependencies from overload cascades during "
            "unexpected traffic spikes."
        ),
    },
    {
        "id": "chunk_20",
        "topic": "distributed_tracing",
        "text": (
            "Distributed tracing propagates a unique trace ID across all service "
            "boundaries using W3C TraceContext headers. Each service creates child "
            "spans annotated with operation name, duration, and key-value attributes. "
            "A tracing backend such as Jaeger or Tempo aggregates spans into a "
            "flame-graph view that exposes serialisation bottlenecks, N+1 query "
            "patterns, and slow external calls invisible in aggregate metrics."
        ),
    },
    {
        "id": "chunk_21",
        "topic": "chaos_engineering",
        "text": (
            "Chaos engineering deliberately injects failures into production or "
            "staging environments to validate resilience assumptions. Experiments "
            "introduce latency, terminate random pods, exhaust file descriptors, or "
            "partition network links between services. A steady-state hypothesis "
            "defines acceptable system behaviour before and after the fault, and the "
            "experiment is aborted automatically if real user impact is detected."
        ),
    },
    {
        "id": "chunk_22",
        "topic": "database_sharding",
        "text": (
            "Horizontal sharding partitions a dataset across multiple database nodes "
            "by hashing or range-bucketing a shard key. Queries that include the shard "
            "key are routed directly to the owning shard, avoiding cross-shard scatter. "
            "Resharding is complex — consistent hashing minimises the fraction of keys "
            "that must migrate when a new shard is added, but cross-shard transactions "
            "still require application-level coordination."
        ),
    },
    {
        "id": "chunk_23",
        "topic": "cqrs",
        "text": (
            "Command Query Responsibility Segregation separates write models from read "
            "models. Commands mutate state and emit events; projections consume those "
            "events to maintain eventually-consistent read replicas optimised for "
            "specific query patterns. This allows independent scaling of the read and "
            "write paths and enables polyglot persistence — a relational write store "
            "alongside a search index and a graph database for different query types."
        ),
    },
    {
        "id": "chunk_24",
        "topic": "columnar_storage",
        "text": (
            "Columnar storage formats such as Parquet and ORC organise data by column "
            "rather than by row, enabling vectorised predicate pushdown that skips "
            "entire row groups irrelevant to a query. Run-length encoding and dictionary "
            "compression exploit column value locality, achieving compression ratios "
            "an order of magnitude better than row-oriented formats for analytical "
            "workloads with high column cardinality."
        ),
    },
    {
        "id": "chunk_25",
        "topic": "stream_processing",
        "text": (
            "Apache Flink and Kafka Streams process unbounded event streams with "
            "exactly-once semantics via distributed snapshots and transactional "
            "Kafka offsets. Windowing operators aggregate events by tumbling, sliding, "
            "or session windows, emitting partial results as watermarks advance. "
            "Stateful operators store keyed state in RocksDB-backed remote storage, "
            "enabling recovery from checkpoints after node failure."
        ),
    },
    {
        "id": "chunk_26",
        "topic": "batch_inference",
        "text": (
            "Batch inference pipelines score large datasets offline using distributed "
            "frameworks such as Ray or Spark. Model artefacts are loaded once per "
            "worker process and reused across partitions to amortise initialisation "
            "cost. GPU-accelerated executors coalesce multiple rows into a single "
            "tensor, saturating hardware throughput. Results are written directly "
            "to a data lake for downstream consumption by analytical queries."
        ),
    },
    {
        "id": "chunk_27",
        "topic": "model_registry",
        "text": (
            "A model registry tracks the full lifecycle of ML artefacts — experiment "
            "runs, hyperparameters, evaluation metrics, and serialised weights. "
            "Versioned model URIs enable reproducible deployments and audit trails. "
            "Stage transitions — staging, champion, archived — gate promotion via "
            "automated evaluation thresholds, ensuring only validated models reach "
            "the production serving layer."
        ),
    },
    {
        "id": "chunk_28",
        "topic": "feature_store",
        "text": (
            "A feature store centralises feature computation, storage, and serving. "
            "The offline store provides point-in-time correct feature snapshots for "
            "training, preventing target leakage by joining features at the event "
            "timestamp rather than the current time. The online store exposes the "
            "same features at sub-millisecond latency during inference, ensuring "
            "training-serving skew is eliminated by design."
        ),
    },
    {
        "id": "chunk_29",
        "topic": "data_drift",
        "text": (
            "Data drift occurs when the statistical distribution of production inputs "
            "diverges from the training distribution. Population Stability Index and "
            "Kolmogorov-Smirnov tests detect covariate drift in numerical features, "
            "while chi-squared tests flag categorical shifts. Automated retraining "
            "pipelines trigger when drift exceeds a threshold, preventing silent "
            "model degradation without requiring human review of every metric."
        ),
    },
    {
        "id": "chunk_30",
        "topic": "attention_mechanism",
        "text": (
            "The self-attention mechanism computes pairwise compatibility scores between "
            "all token positions in a sequence, allowing each token to aggregate context "
            "from the entire input regardless of distance. Scaled dot-product attention "
            "divides logits by the square root of the head dimension before softmax to "
            "prevent gradient vanishing in deep stacks. Multi-head attention runs "
            "several attention functions in parallel, each learning different relational "
            "patterns."
        ),
    },
    {
        "id": "chunk_31",
        "topic": "rag_chunking",
        "text": (
            "Chunking strategy significantly affects retrieval quality in RAG systems. "
            "Fixed-size windows with overlap preserve sentence boundaries but may split "
            "paragraphs mid-thought. Semantic chunking uses embedding similarity to "
            "identify natural topic boundaries, producing variable-length chunks that "
            "align with coherent passages. Recursive character splitting applies "
            "hierarchical separators — paragraph, sentence, word — falling back to "
            "finer granularity when chunks exceed a maximum token budget."
        ),
    },
    {
        "id": "chunk_32",
        "topic": "hyde",
        "text": (
            "Hypothetical Document Embedding generates a synthetic answer to a query "
            "using a language model, then embeds the hypothesis rather than the raw "
            "query. The hypothesis occupies the same semantic space as corpus documents, "
            "bridging the vocabulary gap between short questions and long passages. "
            "HyDE is particularly effective for technical domains where queries lack "
            "the jargon present in the indexed corpus."
        ),
    },
    {
        "id": "chunk_33",
        "topic": "reranking",
        "text": (
            "Cross-encoder rerankers evaluate each query-passage pair jointly, producing "
            "a relevance score orders of magnitude more accurate than bi-encoder "
            "retrieval. Because cross-encoders are computationally expensive they are "
            "applied only to the top-K candidates retrieved by a fast first-stage "
            "retriever. Cohere Rerank, FlashRank, and BGE-Reranker are common choices "
            "that integrate with the retrieve-then-rerank pipeline pattern."
        ),
    },
    {
        "id": "chunk_34",
        "topic": "sparse_retrieval",
        "text": (
            "BM25 is a probabilistic sparse retrieval function that scores documents "
            "by term frequency saturation and inverse document frequency. Unlike dense "
            "retrievers, BM25 requires no training and excels at exact keyword matching, "
            "making it robust to out-of-distribution queries. Hybrid search combines "
            "BM25 and dense retrieval scores via linear interpolation or Reciprocal "
            "Rank Fusion, capturing both lexical and semantic relevance signals."
        ),
    },
    {
        "id": "chunk_35",
        "topic": "graph_rag",
        "text": (
            "Graph-augmented RAG constructs a knowledge graph from corpus entities and "
            "relations extracted by an NLP pipeline. At query time, entity linking "
            "anchors the query to graph nodes, and multi-hop traversal retrieves "
            "supporting facts invisible to flat vector search. The graph community "
            "summaries produced by algorithms such as Leiden clustering provide "
            "global document understanding that complements dense passage retrieval."
        ),
    },
    {
        "id": "chunk_36",
        "topic": "prompt_engineering",
        "text": (
            "Few-shot prompting prepends exemplar input-output pairs to a language "
            "model context, steering generation toward a desired format or reasoning "
            "style. Chain-of-thought prompting elicits step-by-step reasoning before "
            "the final answer, improving accuracy on multi-step arithmetic and "
            "symbolic tasks. Structured output prompting constrains generation to "
            "JSON or XML by combining a schema description with format-enforcing "
            "logit biases or grammar samplers."
        ),
    },
    {
        "id": "chunk_37",
        "topic": "rlhf",
        "text": (
            "Reinforcement Learning from Human Feedback fine-tunes a language model "
            "to align with human preferences. A reward model trained on ranked "
            "pairwise comparisons assigns scalar rewards to model completions. "
            "Proximal Policy Optimisation updates the language model policy to "
            "maximise expected reward while a KL penalty prevents excessive deviation "
            "from the supervised fine-tuned reference policy."
        ),
    },
    {
        "id": "chunk_38",
        "topic": "lora_finetuning",
        "text": (
            "Low-Rank Adaptation inserts trainable rank-decomposition matrices into "
            "each transformer layer, keeping base model weights frozen. This reduces "
            "the number of trainable parameters by orders of magnitude, enabling "
            "fine-tuning on a single GPU. Multiple LoRA adapters can be merged at "
            "inference time by simple matrix addition, allowing task-specific "
            "specialisation without maintaining separate model copies."
        ),
    },
    {
        "id": "chunk_39",
        "topic": "speculative_decoding",
        "text": (
            "Speculative decoding accelerates autoregressive generation by running a "
            "small draft model to propose multiple candidate tokens, which the larger "
            "target model verifies in a single parallel forward pass. Accepted tokens "
            "advance the sequence; rejected tokens cause a rollback to the last "
            "accepted position. On typical language generation tasks, this yields "
            "two-to-four times throughput improvement at identical output quality."
        ),
    },
    {
        "id": "chunk_40",
        "topic": "kv_cache",
        "text": (
            "The key-value cache stores intermediate attention states for previously "
            "processed tokens, allowing autoregressive decoding to avoid recomputing "
            "attention over the full context at each step. Memory footprint grows "
            "linearly with sequence length and batch size. Paged attention manages "
            "KV cache in non-contiguous memory blocks, enabling thousands of concurrent "
            "sequences without memory fragmentation or out-of-memory crashes."
        ),
    },
    {
        "id": "chunk_41",
        "topic": "continuous_batching",
        "text": (
            "Continuous batching, also called iteration-level scheduling, inserts new "
            "requests into a running batch as soon as existing sequences finish "
            "generation, rather than waiting for all sequences in a batch to complete. "
            "This eliminates GPU idle time caused by variable-length outputs and "
            "increases throughput by 20-30x compared to static batching on "
            "transformer serving workloads with diverse sequence lengths."
        ),
    },
    {
        "id": "chunk_42",
        "topic": "tensor_parallelism",
        "text": (
            "Tensor parallelism splits individual weight matrices across multiple GPUs, "
            "partitioning attention heads and feed-forward rows so each device processes "
            "a shard of every token simultaneously. An all-reduce collective gathers "
            "partial sums after each layer. Combined with pipeline parallelism across "
            "model layers, tensor parallelism enables models with hundreds of billions "
            "of parameters to fit within a multi-node GPU cluster."
        ),
    },
    {
        "id": "chunk_43",
        "topic": "flash_attention",
        "text": (
            "FlashAttention rewrites the attention kernel to tile computation across "
            "SRAM, avoiding materialisation of the full N×N attention matrix in "
            "high-bandwidth memory. IO complexity drops from O(N²) to O(N), "
            "enabling efficient attention over sequences of tens of thousands of tokens. "
            "The backward pass recomputes attention weights on the fly during gradient "
            "computation rather than storing them, halving peak memory at the cost "
            "of additional arithmetic."
        ),
    },
    {
        "id": "chunk_44",
        "topic": "mixture_of_experts",
        "text": (
            "Mixture-of-Experts models replace dense feed-forward layers with a set "
            "of expert sub-networks, routing each token to a sparse subset via a "
            "learned gating function. Only two to eight experts are activated per "
            "token regardless of total expert count, keeping FLOPs constant while "
            "growing model capacity. Load-balancing auxiliary losses prevent router "
            "collapse where all tokens route to the same experts."
        ),
    },
    {
        "id": "chunk_45",
        "topic": "embedding_quantisation",
        "text": (
            "Binary and scalar quantisation compress embedding vectors from FP32 to "
            "one or eight bits per dimension, reducing index storage by 4-32× with "
            "a modest recall degradation. Matryoshka Representation Learning trains "
            "embeddings so that any prefix of the full vector retains most information, "
            "enabling a dynamic accuracy-speed tradeoff by truncating at retrieval time "
            "without retraining the model."
        ),
    },
    {
        "id": "chunk_46",
        "topic": "ann_algorithms",
        "text": (
            "Hierarchical Navigable Small World graphs support approximate nearest-"
            "neighbour search with logarithmic query complexity. During construction "
            "each vector is linked to its neighbours at multiple granularity layers; "
            "search enters at the coarsest layer and greedily descends to the query "
            "neighbourhood. HNSW outperforms IVF-based indexes on high-dimensional "
            "dense vectors but requires more memory and slower index build time."
        ),
    },
    {
        "id": "chunk_47",
        "topic": "multimodal_embeddings",
        "text": (
            "Contrastive vision-language models such as CLIP align image and text "
            "representations in a shared embedding space via contrastive loss on "
            "paired image-caption data. Queries expressed as natural language retrieve "
            "images without manual tagging, and images retrieve semantically related "
            "text passages. This enables cross-modal RAG where a user question "
            "retrieves relevant diagrams, charts, or photographs alongside text."
        ),
    },
    {
        "id": "chunk_48",
        "topic": "contextual_compression",
        "text": (
            "Contextual compression extracts only the query-relevant sentences from "
            "retrieved passages before passing them to a language model, reducing "
            "context window consumption and improving answer accuracy. An LLM-based "
            "extractor or an embeddings-based filter scores each sentence against "
            "the query independently, dropping low-relevance content. This is "
            "especially useful when corpus chunks contain multiple distinct topics."
        ),
    },
    {
        "id": "chunk_49",
        "topic": "semantic_cache",
        "text": (
            "Semantic caching stores LLM responses keyed by the embedding of the "
            "query rather than exact string match. An incoming query is embedded and "
            "compared against cached query embeddings; if the nearest neighbour exceeds "
            "a similarity threshold the cached answer is returned immediately. This "
            "cuts latency and cost for repeated or paraphrased questions without "
            "requiring exact duplicates."
        ),
    },
    {
        "id": "chunk_50",
        "topic": "tls_mutual_auth",
        "text": (
            "Mutual TLS authenticates both client and server by exchanging X.509 "
            "certificates during the TLS handshake. Service meshes automate certificate "
            "issuance and rotation via a certificate authority sidecar, eliminating "
            "long-lived credentials. Short-lived certificates — typically valid for "
            "24 hours — limit the blast radius of a compromised private key without "
            "requiring manual revocation."
        ),
    },
    {
        "id": "chunk_51",
        "topic": "zero_trust",
        "text": (
            "Zero-trust architectures enforce identity verification for every request "
            "regardless of network origin, eliminating implicit trust granted by "
            "corporate VPN membership. Policy enforcement points evaluate device "
            "posture, user identity, and resource sensitivity before granting access. "
            "Continuous authorisation re-evaluates sessions against current policy, "
            "revoking access if device compliance or user context changes mid-session."
        ),
    },
    {
        "id": "chunk_52",
        "topic": "secret_management",
        "text": (
            "Secrets managers such as HashiCorp Vault and AWS Secrets Manager store "
            "credentials, API keys, and certificates encrypted at rest and in transit. "
            "Dynamic secrets are generated on demand with a short TTL, scoped to "
            "the requesting service identity. Audit logs capture every secret access "
            "event, enabling forensic analysis after a security incident without "
            "exposing plaintext credentials in application configuration."
        ),
    },
    {
        "id": "chunk_53",
        "topic": "supply_chain_security",
        "text": (
            "Software supply chain security encompasses dependency auditing, image "
            "provenance verification, and build reproducibility. SLSA frameworks grade "
            "build processes on hermeticity and single-source attestation. Sigstore "
            "cosign signs container images and provenance attestations against an "
            "append-only transparency log, allowing consumers to verify that an image "
            "was produced by a specific CI pipeline from a specific source commit."
        ),
    },
    {
        "id": "chunk_54",
        "topic": "ebpf_observability",
        "text": (
            "Extended Berkeley Packet Filter programs run in a sandboxed kernel "
            "virtual machine, attaching to kernel tracepoints and network hooks without "
            "modifying application code. eBPF-based observability tools capture "
            "per-process syscall latency, TCP retransmit rates, and memory allocation "
            "stacks with negligible overhead. Cilium leverages eBPF for network policy "
            "enforcement and service map generation at kernel speed."
        ),
    },
    {
        "id": "chunk_55",
        "topic": "gitops",
        "text": (
            "GitOps treats a Git repository as the single source of truth for "
            "infrastructure and application state. An operator such as ArgoCD or Flux "
            "continuously reconciles the live cluster state against the desired state "
            "declared in Git, applying diffs automatically or after human approval. "
            "Pull-request-based workflows enforce peer review and provide a complete "
            "audit trail for every cluster change."
        ),
    },
    {
        "id": "chunk_56",
        "topic": "infrastructure_as_code",
        "text": (
            "Terraform and Pulumi express cloud infrastructure as declarative code, "
            "enabling version control, code review, and automated testing of "
            "environment changes. State files record the mapping between resource "
            "definitions and live cloud objects, allowing incremental plan-and-apply "
            "cycles. Remote state locking prevents concurrent modifications from "
            "corrupting infrastructure configuration."
        ),
    },
    {
        "id": "chunk_57",
        "topic": "cost_optimisation",
        "text": (
            "Cloud cost optimisation combines right-sizing, spot instance usage, and "
            "reserved capacity commitments. Autoscalers configured with aggressive "
            "scale-in policies reduce idle capacity between traffic waves. Spot "
            "interruption handlers checkpoint running jobs and reschedule them on "
            "fresh instances, enabling batch workloads to run at 60-90% savings "
            "compared to on-demand pricing with minimal impact on completion time."
        ),
    },
    {
        "id": "chunk_58",
        "topic": "multi_region",
        "text": (
            "Multi-region deployments distribute service replicas across geographic "
            "regions to reduce latency for globally distributed users and tolerate "
            "regional outages. Active-active configurations serve writes from the "
            "nearest region using conflict-free replicated data types or last-writer-"
            "wins semantics. Global load balancers route users to the lowest-latency "
            "healthy region using anycast and health-check-weighted DNS."
        ),
    },
    {
        "id": "chunk_59",
        "topic": "disaster_recovery",
        "text": (
            "Recovery Time Objective and Recovery Point Objective define the acceptable "
            "downtime and data loss thresholds for a system. Warm standby deployments "
            "maintain a scaled-down replica in a secondary region with continuous "
            "replication, enabling promotion within minutes. Periodic restore drills "
            "validate backup integrity and recovery runbooks, preventing scenarios "
            "where backups exist but cannot be restored under time pressure."
        ),
    },
    {
        "id": "chunk_60",
        "topic": "api_versioning",
        "text": (
            "API versioning strategies include URI path versioning, header-based "
            "negotiation, and semantic versioning of the schema. Deprecation notices "
            "are communicated via Sunset headers and changelogs with a minimum notice "
            "period. Backward-compatible changes — adding optional fields — require "
            "no version bump, while breaking changes — removing fields or altering "
            "semantics — mandate a new major version with a parallel migration window."
        ),
    },
    {
        "id": "chunk_61",
        "topic": "opentelemetry",
        "text": (
            "OpenTelemetry defines a vendor-neutral API and SDK for emitting traces, "
            "metrics, and logs. Auto-instrumentation agents inject telemetry into "
            "popular frameworks without code changes, while manual instrumentation "
            "adds business context unavailable to generic agents. The OTLP protocol "
            "transmits telemetry to a collector pipeline that enriches, filters, "
            "and exports to multiple backends simultaneously."
        ),
    },
    {
        "id": "chunk_62",
        "topic": "slo_management",
        "text": (
            "Service Level Objectives define the target reliability a service owner "
            "commits to deliver. Error budgets quantify the allowable unreliability "
            "over a rolling window, balancing feature velocity against stability. "
            "When the error budget is nearly exhausted, release freezes and reliability "
            "work take precedence. SLO-based alerting pages on-call engineers only "
            "when the burn rate threatens the budget, reducing alert fatigue from "
            "low-severity transient errors."
        ),
    },
    {
        "id": "chunk_63",
        "topic": "postgres_replication",
        "text": (
            "PostgreSQL streaming replication ships WAL records from the primary to "
            "standby instances in near real time. Synchronous replication waits for "
            "at least one standby to confirm receipt before acknowledging the write "
            "to the client, guaranteeing zero data loss at the cost of added latency. "
            "Logical replication selectively streams row changes for individual tables, "
            "enabling cross-version upgrades and selective data distribution to "
            "downstream consumers."
        ),
    },
    {
        "id": "chunk_64",
        "topic": "connection_pooling",
        "text": (
            "Database connection poolers such as PgBouncer multiplex thousands of "
            "application connections onto a small pool of persistent server connections, "
            "dramatically reducing PostgreSQL backend process overhead. Transaction-mode "
            "pooling recycles connections between statements within a transaction, "
            "enabling higher concurrency at the cost of disabling session-level features "
            "such as advisory locks and prepared statements."
        ),
    },
    {
        "id": "chunk_65",
        "topic": "change_data_capture",
        "text": (
            "Change data capture reads the database write-ahead log to stream row-level "
            "insert, update, and delete events to downstream consumers without polling. "
            "Debezium connectors translate WAL records into Kafka messages keyed by "
            "primary key, enabling cache invalidation, search index synchronisation, "
            "and real-time analytics pipelines that stay consistent with the source "
            "of truth without impacting transactional write performance."
        ),
    },
    {
        "id": "chunk_66",
        "topic": "time_series_db",
        "text": (
            "Time-series databases such as InfluxDB and TimescaleDB are optimised for "
            "append-heavy workloads where data is ordered by timestamp. Automatic data "
            "tiering moves older data to compressed columnar storage, reducing storage "
            "costs while retaining full query access. Continuous aggregates materialise "
            "roll-up summaries in the background, providing millisecond dashboard "
            "queries over months of raw metric data."
        ),
    },
    {
        "id": "chunk_67",
        "topic": "vector_database",
        "text": (
            "Purpose-built vector databases such as Pinecone, Weaviate, and Qdrant "
            "manage embedding storage, indexing, and retrieval behind a simple API. "
            "They support metadata filtering that prunes the search space before ANN "
            "traversal, enabling hybrid semantic-plus-structured queries. Namespace "
            "isolation allows multi-tenant applications to share a single cluster "
            "without cross-tenant data leakage."
        ),
    },
    {
        "id": "chunk_68",
        "topic": "object_storage",
        "text": (
            "Object storage systems such as Amazon S3 and Google Cloud Storage provide "
            "virtually unlimited capacity with 11-nines durability through erasure "
            "coding across availability zones. Lifecycle policies automatically "
            "transition objects to cheaper storage tiers based on age or access "
            "frequency. Presigned URLs grant time-limited direct client access to "
            "private objects without routing through an application server, reducing "
            "bandwidth costs and latency."
        ),
    },
    {
        "id": "chunk_69",
        "topic": "content_delivery_network",
        "text": (
            "Content delivery networks cache static assets at edge PoPs close to "
            "end users, reducing origin server load and improving time-to-first-byte. "
            "Cache-control headers and surrogate keys control invalidation granularity, "
            "enabling targeted purges of related objects without full cache flush. "
            "Edge compute platforms allow lightweight business logic — authentication "
            "checks, A/B routing, header manipulation — to execute at the PoP, "
            "eliminating round trips to origin for dynamic but cacheable decisions."
        ),
    },
    {
        "id": "chunk_70",
        "topic": "dns_failover",
        "text": (
            "DNS-based failover routes traffic away from unhealthy endpoints by "
            "lowering their TTL and removing them from the response set. Health checks "
            "poll endpoints at configurable intervals; successive failures trigger "
            "automatic removal. Short TTLs propagate the change quickly but increase "
            "resolver load; a traffic management service such as Route 53 or Cloud DNS "
            "applies weighted routing policies to facilitate gradual failover."
        ),
    },
    {
        "id": "chunk_71",
        "topic": "websockets",
        "text": (
            "WebSocket connections upgrade from HTTP and maintain a persistent "
            "full-duplex channel, eliminating polling overhead for real-time "
            "applications. Server-sent events provide a simpler unidirectional "
            "alternative for push notifications. Sticky sessions or a shared pub-sub "
            "broker are required when scaling WebSocket servers horizontally, as "
            "messages must reach the connection-holding server regardless of "
            "which backend the client originally connected to."
        ),
    },
    {
        "id": "chunk_72",
        "topic": "graphql",
        "text": (
            "GraphQL exposes a single endpoint that accepts declarative queries "
            "specifying exactly which fields and nested relations to fetch. Clients "
            "avoid over-fetching and under-fetching, reducing payload size and "
            "round-trip count compared to REST. Persisted queries hash the query "
            "document at build time, allowing servers to reject ad-hoc queries "
            "in production and enabling CDN caching of GET-serialised queries."
        ),
    },
    {
        "id": "chunk_73",
        "topic": "protocol_buffers",
        "text": (
            "Protocol Buffers define message schemas in a language-neutral IDL. "
            "The generated binary encoding is compact and fast to serialise, making "
            "it suitable for high-throughput inter-service communication. Field "
            "numbers allow schema evolution — adding optional fields preserves "
            "backward compatibility — while required fields are intentionally absent "
            "from proto3 to prevent parsing failures when fields are removed."
        ),
    },
    {
        "id": "chunk_74",
        "topic": "kafka_architecture",
        "text": (
            "Apache Kafka persists messages on disk in an ordered, immutable log "
            "partitioned across a cluster. Producers append to the partition leader; "
            "consumer groups maintain independent offsets, enabling multiple downstream "
            "systems to replay the same events independently. Tiered storage offloads "
            "cold segments to object storage, decoupling retention period from "
            "broker disk capacity."
        ),
    },
    {
        "id": "chunk_75",
        "topic": "exactly_once_semantics",
        "text": (
            "Kafka exactly-once semantics prevent duplicate message production through "
            "idempotent producers that deduplicate retries using sequence numbers. "
            "Transactional APIs atomically commit messages across multiple partitions "
            "and update consumer offsets in the same Kafka transaction, ensuring "
            "read-process-write pipelines produce each output record exactly once "
            "even under producer crashes or network partitions."
        ),
    },
    {
        "id": "chunk_76",
        "topic": "kubernetes_operators",
        "text": (
            "Kubernetes operators extend the API server with custom resource definitions "
            "and reconciliation loops that encode operational knowledge about stateful "
            "applications. An operator for a distributed database manages provisioning, "
            "failover, backup scheduling, and rolling upgrades by watching CRD objects "
            "and issuing API calls to converge the cluster toward the desired state "
            "expressed in the manifest."
        ),
    },
    {
        "id": "chunk_77",
        "topic": "node_affinity",
        "text": (
            "Node affinity rules constrain which nodes a Kubernetes pod may be scheduled "
            "on, based on node labels representing hardware characteristics such as GPU "
            "type, geographic region, or bare-metal designation. Pod anti-affinity "
            "spreads replicas across nodes or zones to avoid correlated failures. "
            "Topology-spread constraints balance replica count across failure domains "
            "without hard affinity rules that could make pods unschedulable."
        ),
    },
    {
        "id": "chunk_78",
        "topic": "persistent_volumes",
        "text": (
            "Persistent Volumes in Kubernetes decouple storage lifecycle from pod "
            "lifecycle. StorageClasses define provisioner, reclaim policy, and volume "
            "mode. Dynamic provisioning creates cloud disks on demand when a "
            "PersistentVolumeClaim is created, binding it to the pod. ReadWriteOnce "
            "volumes attach to a single node; ReadWriteMany requires a distributed "
            "filesystem such as NFS or a CSI driver that supports concurrent access."
        ),
    },
    {
        "id": "chunk_79",
        "topic": "serverless_functions",
        "text": (
            "Serverless functions execute event-driven workloads without managing "
            "underlying infrastructure. The platform scales from zero to thousands "
            "of concurrent instances within milliseconds, billing only for execution "
            "time. Cold starts — the latency of initialising a new container — are "
            "mitigated by provisioned concurrency that keeps a warm pool of instances "
            "ready. Functions are best suited to short-lived, stateless workloads "
            "with spiky traffic patterns."
        ),
    },
    {
        "id": "chunk_80",
        "topic": "workflow_orchestration",
        "text": (
            "Workflow orchestration engines such as Temporal and Apache Airflow "
            "coordinate multi-step processes with retry policies, timeouts, and "
            "branching logic. Temporal persists workflow history in a durable event "
            "log, enabling workflows to resume transparently after worker restarts. "
            "DAG-based schedulers in Airflow resolve task dependencies and parallelise "
            "independent branches, optimising wall-clock time for data pipeline runs."
        ),
    },
    {
        "id": "chunk_81",
        "topic": "mlops_pipeline",
        "text": (
            "MLOps pipelines automate the path from raw data to production model. "
            "Data validation gates reject schema violations or distribution anomalies "
            "before training begins. Experiment tracking records hyperparameters and "
            "metrics for every run. Automated evaluation compares challenger and "
            "champion models on a held-out dataset, promoting the challenger only when "
            "it exceeds the champion on the primary business metric."
        ),
    },
    {
        "id": "chunk_82",
        "topic": "ab_testing",
        "text": (
            "A/B testing allocates users to control and treatment variants via a "
            "hashing function on a stable identifier such as user ID. Statistical "
            "significance is assessed using two-sample t-tests or Mann-Whitney tests "
            "depending on metric distribution. Sequential testing methods such as "
            "SPRT allow early stopping when results are conclusive, reducing the "
            "opportunity cost of running experiments longer than necessary."
        ),
    },
    {
        "id": "chunk_83",
        "topic": "embedding_fine_tuning",
        "text": (
            "Domain-adaptive fine-tuning of embedding models on in-domain query-passage "
            "pairs improves retrieval quality for specialised corpora. Contrastive "
            "loss with in-batch negatives is the standard training objective; hard "
            "negative mining using BM25 or cross-encoder scores produces negatives "
            "that are semantically close but irrelevant, sharpening decision boundaries. "
            "Evaluation on a held-out retrieval benchmark measures NDCG@10 and "
            "recall@100 before and after fine-tuning."
        ),
    },
    {
        "id": "chunk_84",
        "topic": "colbert",
        "text": (
            "ColBERT decomposes relevance scoring into late interaction between "
            "per-token query and document embeddings. Each document token vector is "
            "stored in the index; at query time, maximum similarity across document "
            "tokens is summed to produce a score. This preserves token-level "
            "expressiveness absent in single-vector bi-encoders while remaining "
            "computationally tractable for large corpora through PLAID-style indexing."
        ),
    },
    {
        "id": "chunk_85",
        "topic": "pgvector",
        "text": (
            "pgvector adds native vector storage and similarity search to PostgreSQL. "
            "Vectors are stored as a first-class column type; HNSW and IVFFlat indexes "
            "accelerate approximate nearest-neighbour queries with configurable recall "
            "tradeoffs. Filtering on standard SQL predicates prunes the candidate set "
            "before ANN traversal, combining relational and semantic queries in a "
            "single statement without a separate vector store."
        ),
    },
    {
        "id": "chunk_86",
        "topic": "document_parsing",
        "text": (
            "PDF and HTML parsing extracts clean text from mixed-format enterprise "
            "documents. Layout-aware parsers such as Unstructured and PyMuPDF preserve "
            "reading order across multi-column layouts, tables, and figures. Table "
            "extraction converts visual grid structures into Markdown or structured "
            "JSON, enabling downstream chunkers to reason about tabular content "
            "without treating each cell as an isolated sentence fragment."
        ),
    },
    {
        "id": "chunk_87",
        "topic": "metadata_filtering",
        "text": (
            "Metadata filtering constrains vector search to a subset of the corpus "
            "matching structured predicates such as document date, source domain, "
            "or access control label. Pre-filtering applies the predicate before ANN "
            "search, reducing the candidate space but potentially returning fewer "
            "than top-K results. Post-filtering applies the predicate after retrieval, "
            "maintaining recall at the cost of over-fetching candidates."
        ),
    },
    {
        "id": "chunk_88",
        "topic": "context_window_management",
        "text": (
            "Context window management selects which retrieved passages and conversation "
            "history to include within the token budget of a language model. "
            "Map-reduce strategies summarise long contexts in parallel before final "
            "synthesis. Lost-in-the-middle research shows models attend most strongly "
            "to content at the beginning and end of the context, motivating placement "
            "of the most relevant passages at extremes rather than the middle."
        ),
    },
    {
        "id": "chunk_89",
        "topic": "guardrails",
        "text": (
            "Guardrail systems intercept LLM inputs and outputs to enforce safety and "
            "policy constraints. Input classifiers detect prompt injection, jailbreaks, "
            "and off-topic requests. Output validators check for personal information "
            "leakage, hallucinated citations, or policy violations before responses "
            "reach users. Structured output validators parse JSON schemas and re-prompt "
            "the model automatically when the output format is malformed."
        ),
    },
    {
        "id": "chunk_90",
        "topic": "llm_evaluation",
        "text": (
            "LLM-as-judge evaluation uses a capable model to score candidate answers "
            "on criteria such as faithfulness, relevance, and completeness. RAGAS "
            "computes answer faithfulness by checking whether each claim in the "
            "response is entailed by the retrieved context. Human evaluation with "
            "inter-annotator agreement scores validates automated metrics before "
            "deploying them as proxies for production quality monitoring."
        ),
    },
    {
        "id": "chunk_91",
        "topic": "memory_management",
        "text": (
            "Conversational memory systems give long-running agents access to past "
            "interactions beyond the immediate context window. Episodic memory "
            "stores full conversation summaries retrievable by semantic similarity. "
            "Working memory maintains a compressed rolling summary updated after each "
            "exchange. External memory backends such as vector databases enable "
            "retrieval of relevant past conversations ranked by semantic relevance "
            "to the current query."
        ),
    },
    {
        "id": "chunk_92",
        "topic": "agent_tool_use",
        "text": (
            "Tool-augmented LLM agents extend generation with the ability to invoke "
            "external functions — web search, code execution, database queries. The "
            "ReAct loop interleaves reasoning traces with action invocations, grounding "
            "conclusions in retrieved evidence. Tool call schemas are expressed as "
            "JSON Schema objects in the system prompt; the model selects and parametrises "
            "tools by emitting structured function call tokens parsed by the host."
        ),
    },
    {
        "id": "chunk_93",
        "topic": "parallel_retrieval",
        "text": (
            "Fan-out retrieval dispatches queries to multiple retrieval sources "
            "concurrently — a dense index, a sparse BM25 engine, a structured SQL "
            "store — and merges results before generation. Async IO or thread pools "
            "minimise total latency to the slowest source rather than the sum of "
            "all sources. Result fusion weights sources by empirically measured "
            "precision on a development query set."
        ),
    },
    {
        "id": "chunk_94",
        "topic": "query_routing",
        "text": (
            "Query routing classifies incoming questions and directs them to the "
            "most appropriate retrieval backend or chain. A classifier trained on "
            "labelled query categories selects between a structured SQL agent, a "
            "vector retriever, and a web-search tool based on query intent. Routing "
            "reduces average latency by avoiding heavyweight retrieval pipelines "
            "for simple lookup questions answerable by exact database queries."
        ),
    },
    {
        "id": "chunk_95",
        "topic": "rag_evaluation_metrics",
        "text": (
            "Retrieval evaluation metrics include Mean Reciprocal Rank, Normalised "
            "Discounted Cumulative Gain, Precision@K, and Recall@K. MRR measures "
            "where the first relevant document appears; NDCG accounts for graded "
            "relevance and rank position jointly. End-to-end RAG evaluation adds "
            "answer faithfulness, which measures whether the generated answer is "
            "supported by the retrieved context, and answer relevance, which measures "
            "alignment with the original query."
        ),
    },
    {
        "id": "chunk_96",
        "topic": "reciprocal_rank_fusion_theory",
        "text": (
            "Reciprocal Rank Fusion is a rank aggregation method that combines multiple "
            "ranked lists without requiring score calibration across systems. The "
            "dampening constant k=60 was empirically selected to diminish the "
            "disproportionate influence of rank-1 positions. RRF consistently "
            "outperforms individual rankers and simple score averaging on TREC "
            "benchmarks, making it a robust default for hybrid search pipelines "
            "that combine dense and sparse signals."
        ),
    },
    {
        "id": "chunk_97",
        "topic": "grounding",
        "text": (
            "Grounding anchors language model responses to verifiable facts retrieved "
            "from authoritative sources. Citations link each generated claim to a "
            "specific passage, enabling readers to verify accuracy independently. "
            "Retrieval-grounded generation measurably reduces hallucination rates "
            "compared to purely parametric generation, particularly for factual, "
            "domain-specific, or time-sensitive questions where model parameters "
            "encode stale or incomplete knowledge."
        ),
    },
    {
        "id": "chunk_98",
        "topic": "sentence_window_retrieval",
        "text": (
            "Sentence-window retrieval indexes individual sentences for precise "
            "retrieval granularity but returns the surrounding paragraph as context "
            "to the language model. This two-level approach combines the precision of "
            "sentence-level similarity matching with the coherence of paragraph-level "
            "context. The window size — typically two or three sentences on either "
            "side — is tuned on a development set to balance precision and context "
            "completeness."
        ),
    },
    {
        "id": "chunk_99",
        "topic": "late_chunking",
        "text": (
            "Late chunking encodes the full document before splitting, allowing "
            "contextualised token representations to inform chunk embeddings rather "
            "than truncating context at chunk boundaries. The full-document embedding "
            "pass is run once; chunk embeddings are derived by mean-pooling the "
            "corresponding token positions. This preserves cross-chunk coreference "
            "resolution and produces more coherent embeddings for chunks that depend "
            "on earlier document context for their meaning."
        ),
    },
]

CHUNK_IDS = [c["id"] for c in CHUNKS]
TEXTS = [c["text"] for c in CHUNKS]