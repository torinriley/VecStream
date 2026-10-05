import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from vecstream.hnsw_index import HNSWIndex


@given(
    dimension=st.integers(min_value=1, max_value=12),
    count=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=30, deadline=None)
def test_random_insert_delete_sequences_preserve_invariants(dimension, count, seed):
    rng = np.random.default_rng(seed)
    index = HNSWIndex(dimension, M=4, ef_construction=16, seed=seed)
    for i in range(count):
        index.add_item(str(i), rng.normal(size=dimension).astype(np.float32))
        index.validate()
    for i in rng.permutation(count)[: count // 2]:
        index.remove_item(str(i))
        index.validate()


operation = st.tuples(
    st.sampled_from(["insert", "update", "delete", "search", "validate"]),
    st.integers(min_value=0, max_value=24),
)


@given(
    dimension=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    operations=st.lists(operation, min_size=1, max_size=80),
)
@settings(max_examples=35, deadline=None)
def test_random_operation_sequences(dimension, seed, operations):
    rng = np.random.default_rng(seed)
    index = HNSWIndex(dimension, M=4, ef_construction=16, seed=seed)
    for action, numeric_id in operations:
        item_id = str(numeric_id)
        if action in {"insert", "update"}:
            index.add_item(item_id, rng.normal(size=dimension).astype(np.float32))
            index.validate()
        elif action == "delete" and item_id in index.nodes:
            index.remove_item(item_id)
            index.validate()
        elif action == "search":
            results = index.search(rng.normal(size=dimension), k=5, ef_search=10)
            assert len(results) <= 5
            assert all(result_id in index.nodes for result_id, _ in results)
            assert all(np.isfinite(score) for _, score in results)
            assert [score for _, score in results] == sorted(
                [score for _, score in results], reverse=True
            )
        else:
            index.validate()


def test_larger_ef_has_nonworse_aggregate_recall():
    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(300, 16)).astype(np.float32)
    queries = rng.normal(size=(40, 16)).astype(np.float32)
    index = HNSWIndex(16, M=8, ef_construction=40, seed=9)
    normalized = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    for i, vector in enumerate(vectors):
        index.add_item(str(i), vector)
    small_recall = large_recall = 0.0
    for query in queries:
        q = query / np.linalg.norm(query)
        truth = {str(i) for i in np.argsort(-(normalized @ q))[:10]}
        small = {item_id for item_id, _ in index.search(query, k=10, ef_search=10)}
        large = {item_id for item_id, _ in index.search(query, k=10, ef_search=80)}
        small_recall += len(truth & small) / 10
        large_recall += len(truth & large) / 10
    assert large_recall >= small_recall
