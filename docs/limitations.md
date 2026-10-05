# Limitations and concurrency contract

- The HNSW graph is in memory and is rebuilt from collection vectors at startup. Graph serialization remains future work; startup includes construction cost.
- Search is approximate. Quality is recall against exact top-k, not generic “accuracy.”
- Metadata predicates are post-filtered. A matching vector outside the ANN candidate set can be missed; results are not exact filtered top-k.
- Deletion removes a node and incident edges. Returned IDs remain correct, but links are not repaired. In the committed Gaussian experiment, recall did not decline monotonically through 20% deletion; rebuilding improved recall from 0.916 to 0.954.
- Objects provide no internal locking. Concurrent read-only searches on an immutable index do not mutate state, but mutation requires exclusive ownership. There is no process-safe writer protocol.
- Checkpoints protect vector-store mutation visibility and can recover one prior generation. They are not multi-collection transactions; old generations are not yet garbage-collected.
- Python heap and adjacency traversal constrain throughput. Threads are not claimed to scale CPU-bound query work.
- The engine is single-machine, with no replication, authentication, distributed query, or transactional multi-client semantics.
- HNSW quality varies with distribution and insertion order. The committed five-order experiment measured recall@10 from 0.904 to 0.922. Clustered and concentrated distribution studies remain unmeasured.
