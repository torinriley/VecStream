# Performance methodology and current evidence

The harness creates one seeded dataset, computes exact top-k, and runs three experiments: an `ef_search` frontier, independently seeded insertion-order shuffles, and cumulative physical deletion followed by rebuild. Dataset and topology/order/deletion RNG streams are separate. Each configuration performs warmups followed by measured single-query searches and records the Git revision and environment.

The 1,000 × 64 frontier shows recall@10 rising from 0.768 to 1.000 as `ef_search` rises from 10 to 160, while QPS falls from 1,902 to 562. The committed files identify the base revision and explicitly record that the working tree contained the Stage Two changes.

Five insertion orders at `ef_search=20` ranged from 0.904 to 0.922 recall@10 (mean 0.9116, population standard deviation 0.0061). This demonstrates insertion-order sensitivity on one small Gaussian dataset, not statistical significance.

Physical deletion produced recall@10 of 0.908, 0.916, 0.916, and 0.916 at 0%, 5%, 10%, and 20% deleted. A rebuild at 20% yielded 0.954. Deletion did not monotonically reduce recall here. The honest conclusion is that physical deletion changes graph quality unpredictably and rebuilding produced a better graph for this run.

## Measurement-driven optimization

The original `_distance` computed both vector norms for every graph-edge evaluation. HNSW construction and search invoke distance inside their hottest traversal loops. The implementation now normalizes once at insertion, converting repeated cosine calculations into dot products while preserving zero-vector behavior. Exact search stacks vectors and uses a vectorized matrix multiply.

A trustworthy before/after number is not reported because the original benchmark did not measure recall at comparable settings. No speedup is claimed. A future comparison should run this identical harness against the pre-change revision on the same hardware.

The next profiling pass should use `cProfile` or `py-spy` at 10k/100k scale. Potential targets—Python set traversal, heap operations, heuristic comparisons, and checkpoint rewrites—must be confirmed before optimization.
