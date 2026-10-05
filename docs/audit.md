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
- Packaging made large ML libraries core dependencies. Core now requires only NumPy; CLI and text embedding are extras.
- Unsupported speed, compression, optimization, and scalability claims were removed.

## Remaining findings

- `CollectionManager`, `IndexManager`, `QueryEngine`, CLI, and legacy client/server overlap. The primary path is `Collection` + store + HNSW; compatibility deletion requires a breaking release.
- The legacy client/server API mismatch was repaired, but the protocol remains a compatibility layer rather than the project's focus.
- HNSW graph persistence, tombstone compaction, generation cleanup, adversarial datasets, insertion-order variance, and memory accounting remain open.
- Old benchmark images/results predate the recall-aware harness and are not current evidence.
