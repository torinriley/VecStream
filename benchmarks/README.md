# ANN benchmarks

`ann_benchmark.py` measures HNSW against vectorized exact cosine top-k. It provides three deliberately narrow experiments:

- `frontier`: sweep `ef_search` and report recall/latency.
- `insertion-order`: index one fixed dataset under independently seeded shuffles.
- `deletion`: measure baseline, cumulative 5%/10%/20% physical deletion, and rebuild.

```bash
python -m benchmarks.ann_benchmark --experiment frontier --vectors 10000 \
  --dim 128 --queries 100 --ef-search 10 20 40 80 160 --output run.json
python -m benchmarks.ann_benchmark --experiment insertion-order --ef-search 20
python -m benchmarks.ann_benchmark --experiment deletion --ef-search 20
```

Dataset, graph-level, insertion-order, and deletion RNG seeds are separate. Each JSON result records the Git revision, Python/NumPy versions, platform, architecture, configuration, and seeds. The three committed 1,000 × 64 results are small measured examples, not claims about other workloads. Full 10k/100k experiments remain manual.
