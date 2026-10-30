import numpy as np
import pytest

from vecstream import Collection, VectorStore


def same(a, b):
    """Same IDs in the same order; scores equal up to float32 matmul rounding."""
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert [i for i, _ in left] == [i for i, _ in right]
        assert [s for _, s in left] == pytest.approx([s for _, s in right], abs=1e-6)


def vectors(count=120, dim=10, seed=0):
    return np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32)


def test_store_batch_matches_individual_exact_searches():
    store = VectorStore()
    data = vectors()
    for i, v in enumerate(data):
        store.add_vector(f"v{i}", v)
    store.add_vector("zero", np.zeros(10))
    queries = vectors(15, seed=1)
    batch = store.search_batch(queries, k=7, threshold=-1.0)
    same(batch, [store.search_similar(q, k=7, threshold=-1.0) for q in queries])
    same(
        store.search_batch(queries, k=3, threshold=0.2),
        [store.search_similar(q, k=3, threshold=0.2) for q in queries],
    )


def test_store_batch_edge_cases():
    store = VectorStore()
    assert store.search_batch([[1.0, 2.0]], k=3) == [[]]
    store.add_vector("a", [1.0, 0.0])
    with pytest.raises(ValueError):
        store.search_batch([[1.0, 0.0]], k=0)
    with pytest.raises(ValueError):
        store.search_batch([[1.0, 0.0], [1.0]], k=1)


@pytest.mark.parametrize("use_hnsw", [True, False])
def test_collection_batch_equals_single_queries(tmp_path, use_hnsw):
    collection = Collection(
        "c", str(tmp_path), use_hnsw, {"M": 8, "ef_construction": 40}, sync=False
    )
    collection.apply(
        [
            {"op": "put", "id": f"v{i}", "vector": v, "metadata": {"g": i % 4}}
            for i, v in enumerate(vectors())
        ]
    )
    queries = vectors(12, seed=3)
    same(
        collection.search_batch(queries, k=5, ef_search=40),
        [collection.search_similar(q, k=5, ef_search=40) for q in queries],
    )
    same(
        collection.search_batch(queries, k=5, ef_search=40, filter_metadata={"g": 2}),
        [
            collection.search_similar(q, k=5, ef_search=40, filter_metadata={"g": 2})
            for q in queries
        ],
    )


def test_collection_batch_is_all_or_nothing_on_bad_input(tmp_path):
    collection = Collection("c", str(tmp_path), True, sync=False)
    collection.add_vector("a", [1.0, 0.0])
    for bad in ([[1.0, 0.0], [1.0]], [[1.0, 0.0], [float("nan"), 0.0]], []):
        with pytest.raises(ValueError):
            collection.search_batch(bad)
    with pytest.raises(ValueError):
        collection.search_batch([[1.0, 0.0]], k=0)


def test_batch_sees_one_consistent_snapshot(tmp_path):
    import threading

    collection = Collection("c", str(tmp_path), False, sync=False)
    for i in range(50):
        collection.add_vector(f"v{i}", [1.0, float(i)])
    stop = threading.Event()

    def churn():
        i = 0
        while not stop.is_set():
            collection.apply(
                [
                    {"op": "put", "id": "pair-a", "vector": [1.0, 1.0]},
                    {"op": "put", "id": "pair-b", "vector": [1.0, 1.0]},
                ]
            )
            collection.apply([{"op": "del", "id": "pair-a"}, {"op": "del", "id": "pair-b"}])
            i += 1

    thread = threading.Thread(target=churn)
    thread.start()
    try:
        for _ in range(60):
            rows = collection.search_batch([[1.0, 1.0]] * 4, k=60, threshold=0.99)
            present = [{i for i, _ in row if i.startswith("pair")} for row in rows]
            assert all(p == present[0] for p in present)  # atomic batches never half-visible
            assert len(present[0]) in (0, 2)
    finally:
        stop.set()
        thread.join()
