"""Reproducible exact-vs-HNSW recall/latency benchmark.

CI example: python benchmarks/ann_benchmark.py --vectors 1000 --queries 25 --dim 64
Full runs are intentionally opt-in; use --vectors 100000 and desired dimensions.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from vecstream import HNSWIndex, VectorStore


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values), p))


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    vectors = rng.normal(size=(args.vectors, args.dim)).astype(np.float32)
    queries = rng.normal(size=(args.queries, args.dim)).astype(np.float32)
    exact = VectorStore()
    index = HNSWIndex(args.dim, M=args.M, ef_construction=args.ef_construction, seed=args.seed)
    started = time.perf_counter()
    for i, vector in enumerate(vectors):
        item_id = str(i)
        exact.add_vector(item_id, vector)
        index.add_item(item_id, vector)
    build_seconds = time.perf_counter() - started
    index.validate()
    truth = [{item_id for item_id, _ in exact.exact_search(q, k=args.k, threshold=-1.0)} for q in queries]
    rows = []
    for ef in args.ef_search:
        for query in queries[: args.warmups]:
            index.search(query, k=args.k, ef_search=ef)
        latencies, recalls = [], []
        for query, expected in zip(queries, truth):
            started = time.perf_counter_ns()
            actual = {item_id for item_id, _ in index.search(query, k=args.k, ef_search=ef)}
            latencies.append((time.perf_counter_ns() - started) / 1e6)
            recalls.append(len(expected & actual) / args.k)
        rows.append({
            "ef_search": ef,
            f"recall@{args.k}": statistics.mean(recalls),
            "p50_ms": percentile(latencies, 50),
            "p95_ms": percentile(latencies, 95),
            "p99_ms": percentile(latencies, 99),
            "qps": 1000.0 / statistics.mean(latencies),
        })
    return {
        "environment": {"platform": platform.platform(), "processor": platform.processor(),
                        "python": sys.version.split()[0], "numpy": np.__version__},
        "configuration": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
            if key != "output"
        },
        "build": {"seconds": build_seconds, "vectors_per_second": args.vectors / build_seconds},
        "search": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
