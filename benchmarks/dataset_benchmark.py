"""Recall/latency/throughput on a real embedding dataset in ann-benchmarks HDF5 format.

Example (GloVe-100, cosine/angular)::

    curl -O http://ann-benchmarks.com/glove-100-angular.hdf5
    python -m benchmarks.dataset_benchmark glove-100-angular.hdf5 --count 100000 \
        --output benchmarks/results/glove-100-angular-100000.json

Requires ``h5py`` (benchmark-only dependency). Ground truth is recomputed exactly
over the chosen prefix of the training set, so any ``--count`` is valid.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from benchmarks.ann_benchmark import environment
from vecstream import HNSWIndex
from vecstream.native import NativeHNSWIndex, native_available


def rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def exact_neighbors(train: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-k, chunked so memory stays bounded."""
    unit_train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    unit_queries = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-12)
    out = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), 64):
        scores = unit_queries[start : start + 64] @ unit_train.T
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
        out[start : start + 64] = np.take_along_axis(part, order, axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160, 320])
    parser.add_argument("--engine", choices=["native", "python"], default="native")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import h5py

    with h5py.File(args.dataset, "r") as handle:
        train = np.asarray(handle["train"][: args.count], dtype=np.float32)
        queries = np.asarray(handle["test"][: args.queries], dtype=np.float32)
        metric = handle.attrs.get("distance", b"unknown")
    if args.engine == "native" and not native_available():
        raise SystemExit("native library not built: python -m vecstream.native build")
    started = time.perf_counter()
    truth = exact_neighbors(train, queries, args.k)
    truth_seconds = time.perf_counter() - started

    engine = NativeHNSWIndex if args.engine == "native" else HNSWIndex
    memory_before = rss_bytes()
    index = engine(train.shape[1], M=args.M, ef_construction=args.ef_construction, seed=1)
    started = time.perf_counter()
    for position, vector in enumerate(train):
        index.add_item(str(position), vector)
        if (position + 1) % 100_000 == 0:
            print(f"built {position + 1:,} in {time.perf_counter() - started:.0f}s", flush=True)
    build_seconds = time.perf_counter() - started
    index_bytes = rss_bytes() - memory_before

    rows = []
    for ef in args.ef_search:
        latencies, hits = [], 0
        for query, expected in zip(queries, truth, strict=True):
            t0 = time.perf_counter_ns()
            found = index.search(query, args.k, ef)
            latencies.append((time.perf_counter_ns() - t0) / 1e6)
            hits += len({int(i) for i, _ in found} & set(expected.tolist()))
        row = {
            "ef_search": ef,
            f"recall@{args.k}": hits / (args.k * len(queries)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "qps_single_thread": 1000.0 / statistics.mean(latencies),
        }
        if hasattr(index, "search_batch"):
            batch = {}
            for threads in sorted({1, 2, 4, os.cpu_count() or 1}):
                t0 = time.perf_counter()
                index.search_batch(queries, args.k, ef, threads=threads)
                batch[str(threads)] = len(queries) / (time.perf_counter() - t0)
            row["batch_qps_by_threads"] = batch
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "environment": environment(),
        "dataset": {
            "file": args.dataset.name,
            "distance": str(metric),
            "dimension": int(train.shape[1]),
        },
        "configuration": {k: v for k, v in vars(args).items() if k not in ("dataset", "output")},
        "results": {
            "vectors": int(len(train)),
            "queries": int(len(queries)),
            "ground_truth_seconds": truth_seconds,
            "build_seconds": build_seconds,
            "inserts_per_second": len(train) / build_seconds,
            "approx_index_memory_mb": index_bytes / 2**20,
            "search": rows,
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
