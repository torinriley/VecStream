# VecStream

VecStream is a compact vector-search engine centered on a from-scratch HNSW implementation. It is an engineering study of graph construction, ANN recall/latency tradeoffs, exact cosine ground truth, numerical behavior, and crash-safe local persistence. It does not wrap FAISS or hnswlib.

The project is deliberately single-machine and small enough that the important algorithm fits in one module.

## What is implemented

- HNSW insertion and query traversal with deterministic seeding
- exponentially distributed levels with multiplier `1 / ln(M)`
- diversity-aware neighbor selection and degree-aware reciprocal pruning
- explicit graph validation via `HNSWIndex.validate()`
- float32 vectors normalized once for HNSW traversal
- vectorized exact cosine search as the recall oracle
- generation-based local checkpoints without pickle
- recovery to the previous complete checkpoint and explicit corruption errors
- property tests for randomized graph mutations
- reproducible frontier, insertion-order, and deletion experiments

## Architecture

```text
Collection
├── BinaryVectorStore: vectors, IDs, metadata, exact search
└── HNSWIndex: normalized vectors, layered graph, ANN search, validation
```

`Collection` composes storage and search. `VectorStore.exact_search` is intentionally separate from approximate `HNSWIndex.search`.

## Example

```python
from vecstream import HNSWIndex, VectorStore

exact = VectorStore()
ann = HNSWIndex(dim=3, M=16, ef_construction=100, seed=42)
for item_id, vector in [("x", [1, 0, 0]), ("y", [0, 1, 0])]:
    exact.add_vector(item_id, vector)
    ann.add_item(item_id, vector)

ann.validate()
print(ann.search([1, 0.1, 0], k=1, ef_search=40))
```

Zero vectors are accepted and have cosine similarity `0.0` to every vector. Empty, non-1-D, wrong-dimension, nonnumeric, NaN, and infinite inputs are rejected. Stored values use float32.

## Measured recall/latency frontier

This is a committed smoke measurement, not a large-scale performance claim: 1,000 Gaussian vectors, 64 dimensions, 50 queries, `M=16`, `ef_construction=100`, seed 42, Python 3.14.3/NumPy 2.5.3 on Apple arm64. Latency is single-query wall time.

| ef_search | recall@10 | p50 ms | p99 ms | QPS |
|---:|---:|---:|---:|---:|
| 10 | 0.768 | 0.522 | 0.666 | 1,902 |
| 20 | 0.908 | 0.746 | 0.857 | 1,327 |
| 40 | 0.990 | 1.106 | 1.189 | 903 |
| 80 | 0.996 | 1.475 | 1.652 | 674 |
| 160 | 1.000 | 1.777 | 1.861 | 562 |

Build time was 7.37 s (136 vectors/s). The result demonstrates the expected frontier: greater search effort recovers more exact neighbors while reducing throughput. Full raw output is in [`benchmarks/results/frontier-1000x64.json`](benchmarks/results/frontier-1000x64.json). Results are hardware- and dataset-specific.

At `ef_search=20`, five deterministic insertion orders produced recall@10 from 0.904 to 0.922 (mean 0.9116, population standard deviation 0.0061). This small experiment demonstrates topology sensitivity; it does not establish statistical significance. After cumulative physical deletion of 20%, recall was 0.916 versus a baseline of 0.908; rebuilding the remaining vectors raised it to 0.954. This dataset did not show monotonic deletion degradation, so none is claimed. The rebuild result still shows that an unrepaired graph and a fresh graph are measurably different. These files record the base Git revision and `working_tree_dirty=true`, because Stage Two measurements were produced before its changes were committed.

```bash
python -m benchmarks.ann_benchmark --experiment frontier --vectors 10000 \
  --dim 128 --queries 100 --ef-search 10 20 40 80 160
python -m benchmarks.ann_benchmark --experiment insertion-order --ef-search 20
python -m benchmarks.ann_benchmark --experiment deletion --ef-search 20
```

Use 100,000 vectors and dimensions 128/384/768 for full experiments; those runs are deliberately excluded from CI.

## Storage correctness

Each mutation writes an immutable `gen-*/` directory containing `manifest.json`, `vectors.npy`, `ids.json`, and `metadata.json`. Files are flushed and fsynced before an atomically replaced `CURRENT` pointer makes the generation visible. The pointer retains the prior generation. On open, VecStream validates version, shape, dtype, IDs, and finite values; it falls back once to the prior complete generation, otherwise raises `CorruptStoreError`. Corruption never becomes an empty database.

## Development

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy hypothesis mypy pytest ruff
ruff check --select E,F,I,B vecstream tests benchmarks/ann_benchmark.py
mypy vecstream
pytest -q
```

VecStream is run directly from the repository and is not published as a Python package.

See [HNSW design](docs/design.md), [limitations](docs/limitations.md), [performance notes](docs/performance.md), and the [repository audit](docs/audit.md).

## Scope and limitations

VecStream is an educational/research-quality local engine, not a production database. The graph is in memory, mutation/search concurrency is not synchronized, metadata filtering is post-filtered, physical deletion can reduce graph quality, and Python graph traversal is the principal scaling boundary. There is no replication, transaction protocol, distributed execution, or multi-process writer support.

## License

MIT
