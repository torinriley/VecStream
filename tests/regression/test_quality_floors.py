"""Deterministic quality gates: recall floors and a hardware-independent cost proxy.

Wall-clock latency is too noisy for unit CI, so cost is tracked as the number of
distance evaluations per query, which moves only when the algorithm changes.
Floors sit a few points below currently measured values; raise them when quality
improves, and investigate (do not lower them) when they trip.

Measured at commit time (800 x 32, M=12, ef_construction=60): recall@10 is
0.872 / 0.962 / 0.996 / 1.0 at ef_search 10 / 20 / 40 / 80 with 237 / 344 / 497 / 666
distance evaluations per query.
"""

import numpy as np
import pytest

from vecstream import Collection, HNSWIndex

K = 10


@pytest.fixture(scope="module")
def corpus():
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(800, 32)).astype(np.float32)
    queries = rng.normal(size=(50, 32)).astype(np.float32)
    index = HNSWIndex(32, M=12, ef_construction=60, seed=5)
    for i, vector in enumerate(vectors):
        index.add_item(str(i), vector)
    return index, vectors, queries


def truth(vectors, ids, query, k=K):
    matrix = vectors[ids] / np.linalg.norm(vectors[ids], axis=1, keepdims=True)
    order = np.argsort(-(matrix @ (query / np.linalg.norm(query))))[:k]
    return {str(ids[j]) for j in order}


def measure(index, vectors, queries, ef, ids=None, **kwargs):
    ids = np.arange(len(vectors)) if ids is None else ids
    calls = [0]
    original = index._normalized_distance

    def counted(a, b):
        calls[0] += 1
        return original(a, b)

    index._normalized_distance = counted
    try:
        hits = 0
        for query in queries:
            found = {i for i, _ in index.search(query, K, ef, **kwargs)}
            hits += len(truth(vectors, ids, query) & found)
    finally:
        del index._normalized_distance
    return hits / (K * len(queries)), calls[0] / len(queries)


@pytest.mark.parametrize("ef,floor", [(10, 0.82), (20, 0.93), (40, 0.97), (80, 0.99)])
def test_recall_floor(corpus, ef, floor):
    index, vectors, queries = corpus
    recall, _ = measure(index, vectors, queries, ef)
    assert recall >= floor, f"recall@{K} {recall:.3f} fell below {floor} at ef_search={ef}"


@pytest.mark.parametrize("ef,ceiling", [(10, 320), (20, 460), (40, 660), (80, 880)])
def test_distance_evaluations_per_query_ceiling(corpus, ef, ceiling):
    index, vectors, queries = corpus
    _, evaluations = measure(index, vectors, queries, ef)
    assert evaluations <= ceiling, f"{evaluations:.0f} distance evaluations/query > {ceiling}"


def test_recall_increases_with_search_effort(corpus):
    index, vectors, queries = corpus
    values = [measure(index, vectors, queries, ef)[0] for ef in (10, 20, 40, 80)]
    assert values == sorted(values)


def test_recall_after_deleting_a_quarter_stays_near_baseline(corpus):
    _, vectors, queries = corpus
    index = HNSWIndex(32, M=12, ef_construction=60, seed=5)
    for i, vector in enumerate(vectors):
        index.add_item(str(i), vector)
    doomed = np.random.default_rng(1).permutation(800)[:200]
    for i in doomed:
        index.remove_item(str(int(i)))
    live = np.setdiff1d(np.arange(800), doomed)
    recall, _ = measure(index, vectors, queries, 40, ids=live)
    assert recall >= 0.95


def test_filtered_recall_floor(corpus):
    index, vectors, queries = corpus
    live = np.arange(0, 800, 5)  # 20% selectivity
    allowed = {str(i) for i in live}
    recall, _ = measure(index, vectors, queries, 40, ids=live, predicate=allowed.__contains__)
    assert recall >= 0.95


def test_persisted_graph_returns_identical_results(corpus, tmp_path):
    _, vectors, queries = corpus
    collection = Collection("q", str(tmp_path), True, {"M": 12, "ef_construction": 60}, sync=False)
    collection.apply(
        [{"op": "put", "id": str(i), "vector": v} for i, v in enumerate(vectors[:300])]
    )
    before = [collection.search_similar(q, K, ef_search=40) for q in queries[:15]]
    collection.close()
    reopened = Collection("q", str(tmp_path), True, {"M": 12, "ef_construction": 60}, sync=False)
    assert reopened.recovery["graph_restored"]
    assert [reopened.search_similar(q, K, ef_search=40) for q in queries[:15]] == before
