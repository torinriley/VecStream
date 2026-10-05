# Performance methodology and current evidence

The harness creates one seeded dataset, builds exact and HNSW stores, computes exact top-k once, then sweeps `ef_search`. Each configuration performs warmups followed by measured single-query searches. It records the environment, dataset, seed, HNSW parameters, build throughput, recall@k, latency percentiles, and QPS.

The committed 1,000 × 64 smoke run shows recall@10 rising from 0.758 to 1.000 as `ef_search` rises from 10 to 160, while QPS falls from 1,731 to 505. This is a regression signal, not a 100k-scale claim.

## Measurement-driven optimization

The original `_distance` computed both vector norms for every graph-edge evaluation. HNSW construction and search invoke distance inside their hottest traversal loops. The implementation now normalizes once at insertion, converting repeated cosine calculations into dot products while preserving zero-vector behavior. Exact search stacks vectors and uses a vectorized matrix multiply.

A trustworthy before/after number is not reported because the original benchmark did not measure recall at comparable settings. No speedup is claimed. A future comparison should run this identical harness against the pre-change revision on the same hardware.

The next profiling pass should use `cProfile` or `py-spy` at 10k/100k scale. Potential targets—Python set traversal, heap operations, heuristic comparisons, and checkpoint rewrites—must be confirmed before optimization.
