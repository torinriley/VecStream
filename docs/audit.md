# Repository audit

The initial repository had a useful narrow core but its claims exceeded its evidence.

## Corrected in this pass

- Level generation used `M / M_max0` (always 0.5) rather than `1 / ln(M)`.
- Neighbor selection and pruning retained only nearest candidates, omitting graph diversity.
- Cosine distance recomputed norms throughout traversal; HNSW vectors are now pre-normalized.
- ID updates left stale graph geometry; updates now remove and reinsert.
- Numerical and dimensional validation was incomplete.
- Exact search looped in Python and repeatedly normalized vectors; it is now a vectorized oracle.
- Persistence pickled a dictionary, overwrote files, and suppressed every exception. It now uses versioned immutable generations and explicit errors.
- There was no invariant checker, deterministic seed, statistical level test, property test, or ANN recall benchmark.
- Packaging made large ML libraries core dependencies. Core now requires only NumPy.
- Unsupported speed, compression, optimization, and scalability claims were removed.

## Stage Two cleanup

- Removed `IndexManager`, `QueryEngine`, `PersistentVectorStore`, client/server, CLI, text-embedding dependencies, and their compatibility exports.
- Removed obsolete examples, stress scripts, benchmark code, generated plots, and historical results.
- The only supported path is now `Collection` composed from `BinaryVectorStore` and `HNSWIndex`; `VectorStore` supplies exact-search ground truth.

## Stage Two evidence

- Stage Two added measured insertion-order variance and physical-deletion/rebuild experiments using exact ground truth.

## Remaining findings

- HNSW graph persistence, tombstone compaction, generation cleanup, adversarial datasets, and memory accounting remain open.
