# ANN benchmarks

`ann_benchmark.py` measures VecStream HNSW results against vectorized exact cosine top-k. It reports environment and configuration, build throughput, recall@k, latency percentiles, and QPS while sweeping `ef_search`.

```bash
python benchmarks/ann_benchmark.py --vectors 10000 --dim 128 --queries 100 \
  --ef-search 10 20 40 80 160 --seed 42 --output run.json
```

The committed `results/smoke-1000x64-seed42.json` is a small measured regression artifact. It must not be generalized to other hardware, dimensions, distributions, or dataset sizes. Full 10k/100k experiments are manual because this pure-Python implementation builds too slowly for routine CI at those sizes.

Historical graph images in this directory predate the exact-recall harness and are retained only as project history; they are not evidence for current performance claims.
