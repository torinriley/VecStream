# Limitations and concurrency contract

- Durability: every committed mutation is in the checksummed write-ahead log (fsynced by default) before it is acknowledged. Checkpoints (vectors, metadata, and the HNSW graph) run automatically and on shutdown; startup restores the graph and replays the log instead of rebuilding.
- Concurrency: a `Collection` is safe for many concurrent searches and serialized mutations (writer-preferring readers-writer lock). A checkpoint holds the write lock, so it pauses searches for its duration (proportional to collection size). A bare `HNSWIndex`, `VectorStore`, or `BinaryVectorStore` is not thread-safe; use `Collection`.
- Atomicity scope: a batch is atomic within one collection. There are no transactions across collections.
- One process per storage directory: a `flock` rejects a second opener. There is no multi-process writer protocol, replication, or distributed query.
- WAL damage: a torn final record is discarded (it was never acknowledged). Damage before the tail raises `WALCorruptionError`; `wal_corruption="truncate"` opts in to discarding everything after it.
- Checkpoint fallback: if the newest generation fails its checksums, opening is refused with `CorruptStoreError`, because the previous generation predates acknowledged operations. `allow_stale_checkpoint=True` explicitly accepts that loss; log records that no longer connect to the older checkpoint are discarded rather than replayed onto inconsistent state.
- Search is approximate. Quality is recall against exact top-k, not generic “accuracy.”
- Filtered search enforces metadata predicates and thresholds during traversal, so a result set is filled whenever enough matches exist. It is still approximate for moderately selective filters (nodes are navigated through regardless of the filter; matches reachable only through distant non-matches can be missed). If `max(2000, 50*ef)` nodes are visited without finding `k` matches, the search falls back to an exact scan of all vectors, which is O(n) in Python. Metadata is not indexed; each visited node costs one predicate evaluation.
- Deletion removes a node and repairs its former neighbors' edges (refilling free degree from the removed node's other neighbors by the diversity heuristic). The committed experiment (1,000 x 64) measured recall@10 0.952 after deleting 20% versus 0.954 for a rebuild, down from 0.916 without repair. Repair is local: it does not rebalance upper layers or compact memory. Heavy churn still benefits from an occasional rebuild.
- Engines: the optional native engine (Rust, `native/`, built with `python -m vecstream.native build`) is selected automatically (`engine="auto"`) when present; otherwise the pure-Python engine runs. Graph checkpoints are engine-agnostic. The native engine keeps its own normalized copy of every vector, so index memory is roughly 2x the vector bytes plus the Python-side vector store. It supports the same filters, thresholds, and deletion repair; ID-to-slot mapping stays in Python.
- Parallelism: native searches release the GIL, so concurrent searches from threads and `search_batch` (OS threads inside the library) scale across cores. The pure-Python engine does not: with 4 threads on 4 cores, 200 queries took 6.8s versus 1.2s sequentially, so it runs queries sequentially. Filtered native search calls a Python predicate per visited node, which re-acquires the GIL and limits its scaling. Mutations are serialized per collection and block searches while they run.
- Native safety: the library is a C ABI that validates pointers, sizes, and finiteness, catches panics, and rejects corrupt persisted graphs (the collection then rebuilds). It is `unsafe` at the FFI boundary by nature: use only the shipped wrapper. SIMD is AVX2+FMA on x86-64 (runtime-detected) with a portable auto-vectorized fallback elsewhere; there is no hand-written NEON path.
- Memory-mapped storage, sharding, replication, product quantization, and binary quantization are not implemented. 8-bit scalar quantization is (native engine only; see the quantization measurement below).
- The HTTP server is a threaded standard-library server: adequate for modest load, not tuned for high connection counts. Rate limiting, quotas, audit logging, and encryption at rest are not implemented; TLS is available via `--tls-cert/--tls-key` (or terminate it at a proxy).
- Tenant isolation is by separate directories and credentials inside one process and one filesystem trust domain; it is not a defense against a compromised host or process.
- HNSW quality varies with distribution and insertion order. The committed five-order experiment measured recall@10 from 0.904 to 0.922. Clustered and concentrated distribution studies remain unmeasured.
- Large-scale measurement (native engine, 100,000 x 128 i.i.d. Gaussian vectors, M=16, ef_construction=100, 4 cores; `benchmarks/results/native-frontier-100000x128.json`): build 80s (1,254 inserts/s); recall@10 0.16 / 0.28 / 0.46 / 0.66 at ef_search 40 / 80 / 160 / 320; p50 0.54-1.73 ms; batch throughput 559 / 1,250 / 2,381 qps on 1 / 2 / 4 threads. I.i.d. Gaussian data has no cluster structure and is close to the worst case for graph ANN, so these recall numbers are a floor, not a prediction for real embeddings. Realistic embedding distributions and 1M+ vectors remain unmeasured.
- Real-embedding measurements (native engine, GloVe-100 angular from ann-benchmarks, M=16, ef_construction=100, 4 cores, exact ground truth recomputed per prefix; `benchmarks/dataset_benchmark.py`, results in `benchmarks/results/glove-100-angular-*.json`):

  | Vectors | ef_search | recall@10 | p50 / p99 ms | batch qps (1 / 4 threads) |
  | --- | --- | --- | --- | --- |
  | 100,000 | 80 | 0.868 | 0.38 / 0.73 | 2,845 / 9,959 |
  | 100,000 | 160 | 0.926 | 0.66 / 1.17 | 1,672 / 6,398 |
  | 100,000 | 320 | 0.966 | 1.27 / 2.82 | 704 / 2,449 |
  | 1,183,514 | 160 | 0.854 | 1.10 / 1.79 | 928 / 3,702 |
  | 1,183,514 | 320 | 0.905 | 1.94 / 4.19 | 418 / 1,841 |
  | 1,183,514 | 640 | 0.942 | 4.23 / 7.58 | 260 / 1,016 |

  Full 1.18M build: 1,153s (1,026 inserts/s), about 830 MB of additional resident memory for the index. Reaching 0.95 recall at 1M requires roughly ef_search 700+ at this M; raising M and ef_construction would trade build time and memory for recall and has not been tuned. Single runs on one machine: expect variance, and treat latencies as indicative.
- Index tuning (native engine, GloVe-100 angular, single runs, 4 cores; `benchmarks/results/tuning/`). Raising `M` is the effective lever; raising `ef_construction` past 200 is not:

  | Vectors | M / ef_construction | Build | Extra memory | recall@10 at ef_search 80 / 160 / 320 | p50 ms at ef_search 160 |
  | --- | --- | --- | --- | --- | --- |
  | 100,000 | 16 / 100 | 70s | 87 MB | 0.868 / 0.926 / 0.966 | 0.66 |
  | 100,000 | 16 / 200 | 72s | 87 MB | 0.873 / 0.935 / 0.972 | 0.75 |
  | 100,000 | 24 / 200 | 113s | 92 MB | 0.916 / 0.963 / 0.988 | 0.75 |
  | 100,000 | 32 / 200 | 155s | 96 MB | 0.936 / 0.975 / 0.994 | 0.79 |
  | 100,000 | 32 / 400 | 218s | 96 MB | 0.941 / 0.980 / 0.996 | 0.83 |
  | 100,000 | 48 / 400 | 364s | 105 MB | 0.962 / 0.988 / 0.998 | 0.98 |
  | 1,183,514 | 16 / 100 | 1,153s | 829 MB | 0.786 / 0.854 / 0.905 | 1.10 |
  | 1,183,514 | 32 / 200 | 3,284s | 992 MB | 0.875 / 0.925 / 0.960 | 1.78 |

  At 1.18M vectors M=32 reaches ~0.96 recall at ef_search 320 (2.7 ms p50) where M=16 needs ef_search 640 for 0.94 (4.2 ms), i.e. roughly half the latency at equal recall, in exchange for ~2.9x the build time (part of that is ef_construction 200 vs 100) and ~20% more memory. New collections therefore default to `M=32, ef_construction=200`. Collections that already saved their parameters keep them; keys missing from old saved parameters keep their old meaning (`M=16`). Changing a collection's parameters makes it rebuild its graph on the next open. These are single runs on one machine and one dataset: expect variance, and re-tune for your own data.
- 8-bit scalar quantization (`hnsw_params: {"quantization": "sq8"}`, native engine only; the Python engine raises `ValueError` rather than ignoring it). The index stores one byte per dimension plus a 12-byte header instead of four bytes per dimension, searches fetch `4*k` candidates and re-rank them with exact cosine scores from the full-precision vectors, so returned scores are exact; only candidate selection is approximate. Measured on GloVe-100 angular, 100,000 vectors, M=32, ef_construction=200, 4 cores, single run (`benchmarks/results/tuning/quantization-*-glove-100000.json`); latency includes the Python re-ranking step:

  | ef_search | recall@10 float / sq8 | p50 ms float / sq8 | p99 ms float / sq8 |
  | --- | --- | --- | --- |
  | 40 | 0.876 / 0.875 | 0.31 / 0.31 | 0.71 / 0.54 |
  | 80 | 0.936 / 0.936 | 0.55 / 0.56 | 0.94 / 0.96 |
  | 160 | 0.975 / 0.975 | 0.94 / 0.84 | 1.87 / 1.62 |
  | 320 | 0.994 / 0.994 | 1.66 / 1.36 | 3.19 / 2.27 |

  Index vector storage fell from 38.1 MB to 10.7 MB (3.6x), and the index's resident growth from 96 MB to 59 MB; build time was unchanged (162s vs 153s). Recall at equal `ef_search` was unchanged within noise, because the oversampled candidates are re-ranked exactly. Quantized searches use at least `ef_search = 4*k` (so `ef_search` 20 behaves as 40 at `k=10`; the sq8 run's first row is therefore not a separate point). Batch throughput was mixed rather than better (4 threads: 8,301 / 4,737 / 3,048 qps for float versus 6,560 / 5,538 / 2,557 for sq8 at ef_search 80 / 160 / 320), since each batch pays for 4x the candidates and Python re-ranking. Limits: a `Collection` still keeps the full-precision vectors in its Python store (re-ranking and checkpoints need them), so total process memory falls only by the index's own copy, about 290 bytes per 100-dimensional vector; the saving grows with dimension. With a similarity threshold, the traversal bound is relaxed by 0.02 to allow for quantization error before exact scores decide. One dataset, one machine, one run: re-measure on your own embeddings.
