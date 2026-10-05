"""Three reproducible ANN experiments: frontier, insertion order, deletion."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from vecstream import HNSWIndex, VectorStore


def environment() -> dict:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
        ).stdout.strip()
        working_tree_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                check=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        working_tree_dirty = None
    return {
        "vecstream_revision": revision,
        "working_tree_dirty": working_tree_dirty,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def dataset(count: int, dimension: int, queries: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(count, dimension)).astype(np.float32),
        rng.normal(size=(queries, dimension)).astype(np.float32),
    )


def exact_truth(vectors: dict[str, np.ndarray], queries: np.ndarray, k: int) -> list[set[str]]:
    store = VectorStore()
    for item_id, vector in vectors.items():
        store.add_vector(item_id, vector)
    return [
        {item_id for item_id, _ in store.exact_search(query, k=k, threshold=-1.0)}
        for query in queries
    ]


def evaluate(
    index: HNSWIndex,
    queries: np.ndarray,
    truth: list[set[str]],
    k: int,
    ef_search: int,
    warmups: int,
) -> dict:
    for query in queries[:warmups]:
        index.search(query, k=k, ef_search=ef_search)
    latencies: list[float] = []
    recalls: list[float] = []
    for query, expected in zip(queries, truth, strict=True):
        started = time.perf_counter_ns()
        actual = {item_id for item_id, _ in index.search(query, k=k, ef_search=ef_search)}
        latencies.append((time.perf_counter_ns() - started) / 1e6)
        recalls.append(len(expected & actual) / min(k, len(expected)))
    return {
        f"recall@{k}": statistics.mean(recalls),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "qps": 1000.0 / statistics.mean(latencies),
    }


def build(
    vectors: np.ndarray, order: np.ndarray, args: argparse.Namespace, graph_seed: int
) -> tuple[HNSWIndex, float]:
    index = HNSWIndex(args.dim, M=args.M, ef_construction=args.ef_construction, seed=graph_seed)
    started = time.perf_counter()
    for position in order:
        index.add_item(str(int(position)), vectors[position])
    elapsed = time.perf_counter() - started
    index.validate()
    return index, elapsed


def frontier(args: argparse.Namespace, vectors: np.ndarray, queries: np.ndarray) -> dict:
    live = {str(i): vector for i, vector in enumerate(vectors)}
    truth = exact_truth(live, queries, args.k)
    index, build_seconds = build(vectors, np.arange(len(vectors)), args, args.graph_seed)
    return {
        "build": {"seconds": build_seconds, "vectors_per_second": len(vectors) / build_seconds},
        "search": [
            {"ef_search": ef, **evaluate(index, queries, truth, args.k, ef, args.warmups)}
            for ef in args.ef_search
        ],
    }


def insertion_order(args: argparse.Namespace, vectors: np.ndarray, queries: np.ndarray) -> dict:
    truth = exact_truth({str(i): vector for i, vector in enumerate(vectors)}, queries, args.k)
    rows = []
    ef = args.ef_search[0]
    for order_seed in args.order_seeds:
        order = np.random.default_rng(order_seed).permutation(len(vectors))
        index, build_seconds = build(vectors, order, args, args.graph_seed)
        rows.append(
            {
                "insertion_order_seed": order_seed,
                "build_seconds": build_seconds,
                **evaluate(index, queries, truth, args.k, ef, args.warmups),
            }
        )
    recalls = [row[f"recall@{args.k}"] for row in rows]
    return {
        "ef_search": ef,
        "runs": rows,
        "summary": {
            "mean_recall": statistics.mean(recalls),
            "recall_stddev": statistics.pstdev(recalls),
            "min_recall": min(recalls),
            "max_recall": max(recalls),
        },
    }


def deletion(args: argparse.Namespace, vectors: np.ndarray, queries: np.ndarray) -> dict:
    index, build_seconds = build(vectors, np.arange(len(vectors)), args, args.graph_seed)
    live = {str(i): vector for i, vector in enumerate(vectors)}
    deletion_order = np.random.default_rng(args.deletion_seed).permutation(len(vectors))
    rows = []
    prior_target = 0
    for fraction in [0.0, 0.05, 0.10, 0.20]:
        target = int(len(vectors) * fraction)
        for position in deletion_order[prior_target:target]:
            item_id = str(int(position))
            index.remove_item(item_id)
            del live[item_id]
        prior_target = target
        index.validate()
        truth = exact_truth(live, queries, args.k)
        rows.append(
            {
                "deleted_fraction": fraction,
                "remaining_vectors": len(live),
                **evaluate(index, queries, truth, args.k, args.ef_search[0], args.warmups),
            }
        )
    rebuilt = HNSWIndex(
        args.dim, M=args.M, ef_construction=args.ef_construction, seed=args.graph_seed
    )
    started = time.perf_counter()
    for item_id, vector in live.items():
        rebuilt.add_item(item_id, vector)
    rebuild_seconds = time.perf_counter() - started
    rebuilt.validate()
    truth = exact_truth(live, queries, args.k)
    rows.append(
        {
            "deleted_fraction": 0.20,
            "remaining_vectors": len(live),
            "state": "rebuilt",
            **evaluate(rebuilt, queries, truth, args.k, args.ef_search[0], args.warmups),
        }
    )
    return {
        "initial_build_seconds": build_seconds,
        "rebuild_seconds": rebuild_seconds,
        "ef_search": args.ef_search[0],
        "stages": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=["frontier", "insertion-order", "deletion"], default="frontier"
    )
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--ef-search", type=int, nargs="+", default=[10, 20, 40, 80, 160])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--graph-seed", type=int, default=43)
    parser.add_argument("--order-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--deletion-seed", type=int, default=44)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    vectors, queries = dataset(args.vectors, args.dim, args.queries, args.dataset_seed)
    runner = {"frontier": frontier, "insertion-order": insertion_order, "deletion": deletion}[
        args.experiment
    ]
    result = {
        "environment": environment(),
        "experiment": args.experiment,
        "configuration": {key: value for key, value in vars(args).items() if key != "output"},
        "results": runner(args, vectors, queries),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
