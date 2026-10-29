# Limitations and concurrency contract

- Durability: every committed mutation is in the checksummed write-ahead log (fsynced by default) before it is acknowledged. Checkpoints (vectors, metadata, and the HNSW graph) run automatically and on shutdown; startup restores the graph and replays the log instead of rebuilding.
- Concurrency: a `Collection` is safe for many concurrent searches and serialized mutations (writer-preferring readers-writer lock). A checkpoint holds the write lock, so it pauses searches for its duration (proportional to collection size). A bare `HNSWIndex`, `VectorStore`, or `BinaryVectorStore` is not thread-safe; use `Collection`.
- Atomicity scope: a batch is atomic within one collection. There are no transactions across collections.
- One process per storage directory: a `flock` rejects a second opener. There is no multi-process writer protocol, replication, or distributed query.
- WAL damage: a torn final record is discarded (it was never acknowledged). Damage before the tail raises `WALCorruptionError`; `wal_corruption="truncate"` opts in to discarding everything after it.
- Checkpoint fallback: if the newest generation fails its checksums, opening is refused with `CorruptStoreError`, because the previous generation predates acknowledged operations. `allow_stale_checkpoint=True` explicitly accepts that loss; log records that no longer connect to the older checkpoint are discarded rather than replayed onto inconsistent state.
- Search is approximate. Quality is recall against exact top-k, not generic “accuracy.”
- Metadata predicates are post-filtered. A matching vector outside the ANN candidate set can be missed; results are not exact filtered top-k.
- Deletion removes a node and incident edges. Returned IDs remain correct, but links are not repaired. In the committed Gaussian experiment, recall did not decline monotonically through 20% deletion; rebuilding improved recall from 0.916 to 0.954. Removing a node also rescans for the entry point (O(n)).
- Python heap and adjacency traversal constrain throughput, and the GIL prevents CPU-bound query scaling across threads. Native search, SIMD, quantization, sharding, and replication are not implemented.
- The HTTP server is a threaded standard-library server: adequate for modest load, not tuned for high connection counts. Rate limiting, quotas, audit logging, and encryption at rest are not implemented; TLS is available via `--tls-cert/--tls-key` (or terminate it at a proxy).
- Tenant isolation is by separate directories and credentials inside one process and one filesystem trust domain; it is not a defense against a compromised host or process.
- HNSW quality varies with distribution and insertion order. The committed five-order experiment measured recall@10 from 0.904 to 0.922. Clustered and concentrated distribution studies remain unmeasured.
