"""Predicate / min-score constrained traversal."""

import numpy as np
import pytest

from vecstream import HNSWIndex


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(800, 24)).astype(np.float32)
    index = HNSWIndex(24, M=12, ef_construction=80, seed=3)
    for i, vector in enumerate(vectors):
        index.add_item(str(i), vector)
    return index, vectors


def exact(vectors, query, k, allowed=None, min_score=None):
    matrix = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = matrix @ (query / np.linalg.norm(query))
    order = np.argsort(-scores, kind="stable")
    rows = [
        (str(i), float(scores[i]))
        for i in order
        if (allowed is None or str(i) in allowed) and (min_score is None or scores[i] >= min_score)
    ]
    return rows[:k]


def mean_recall(index, vectors, allowed, queries, k=10, ef=60, **kwargs):
    hits = 0
    for query in queries:
        truth = {i for i, _ in exact(vectors, query, k, allowed)}
        found = {i for i, _ in index.search(query, k, ef, predicate=allowed.__contains__, **kwargs)}
        hits += len(truth & found)
    return hits / (k * len(queries))


QUERIES = np.random.default_rng(21).normal(size=(40, 24)).astype(np.float32)


@pytest.mark.parametrize("selectivity", [0.5, 0.1, 0.03])
def test_results_always_satisfy_predicate_and_recall_is_high(dataset, selectivity):
    index, vectors = dataset
    allowed = {str(i) for i in range(800) if (i * 2654435761) % 1000 < selectivity * 1000}
    for query in QUERIES[:10]:
        results = index.search(query, 10, 60, predicate=allowed.__contains__)
        assert len(results) == min(10, len(allowed))
        assert all(item in allowed for item, _ in results)
        assert [s for _, s in results] == sorted((s for _, s in results), reverse=True)
    assert mean_recall(index, vectors, allowed, QUERIES) >= 0.85


def test_scores_match_exact_cosine(dataset):
    index, vectors = dataset
    allowed = {str(i) for i in range(0, 800, 4)}
    for item, score in index.search(QUERIES[0], 5, 80, predicate=allowed.__contains__):
        expected = dict(exact(vectors, QUERIES[0], 800, allowed))[item]
        assert score == pytest.approx(expected, abs=1e-5)


def test_min_score_alone_returns_only_qualifying_results(dataset):
    index, vectors = dataset
    query = vectors[5]
    results = index.search(query, 50, 120, min_score=0.3)
    assert results and all(score >= 0.3 - 1e-6 for _, score in results)
    truth = {i for i, _ in exact(vectors, query, 50, min_score=0.3)}
    assert len(truth & {i for i, _ in results}) >= 0.9 * len(truth)


def test_min_score_above_every_score_returns_nothing(dataset):
    index, _ = dataset
    assert index.search(QUERIES[0], 5, 40, min_score=1.01) == []


def test_predicate_and_min_score_combine(dataset):
    index, vectors = dataset
    allowed = {str(i) for i in range(0, 800, 2)}
    results = index.search(vectors[10], 20, 100, predicate=allowed.__contains__, min_score=0.2)
    assert all(item in allowed and score >= 0.2 - 1e-6 for item, score in results)


def test_unconstrained_search_is_unchanged_by_the_new_parameters(dataset):
    index, _ = dataset
    plain = index.search(QUERIES[0], 10, 50)
    assert plain == index.search(QUERIES[0], 10, 50, predicate=None, min_score=None)
    assert plain == index.search(QUERIES[0], 10, 50, predicate=lambda _: True)


def test_very_selective_predicate_falls_back_to_exact_scan(dataset):
    index, vectors = dataset
    far = max(range(800), key=lambda i: -float(vectors[i] @ QUERIES[0]))  # least similar vector
    q = QUERIES[0] / np.linalg.norm(QUERIES[0])
    results, visited, fallback = index._search_filtered(
        q, index.ep, 1, 10, lambda item: item == str(far), None, max_visits=100
    )
    assert fallback is True
    assert results[0][0] == str(far)


def test_fallback_is_exact_for_a_sparse_matching_set(dataset):
    index, vectors = dataset
    allowed = {str(i) for i in (3, 77, 401, 650)}
    results = index.search(QUERIES[1], 3, 20, predicate=allowed.__contains__, max_visits=50)
    assert [i for i, _ in results] == [i for i, _ in exact(vectors, QUERIES[1], 3, allowed)]


def test_no_fallback_when_enough_matches_are_found_quickly(dataset):
    index, _ = dataset
    q = QUERIES[2] / np.linalg.norm(QUERIES[2])
    results, visited, fallback = index._search_filtered(
        q, index.ep, 10, 40, lambda _: True, None, max_visits=None
    )
    assert fallback is False and len(results) == 10 and visited < 800


def test_predicate_matching_nothing_returns_empty_without_error(dataset):
    index, _ = dataset
    assert index.search(QUERIES[0], 5, 20, predicate=lambda _: False) == []


def test_predicate_is_evaluated_at_most_once_per_node(dataset):
    index, _ = dataset
    calls = []

    def predicate(item):
        calls.append(item)
        return int(item) % 5 == 0

    index.search(QUERIES[3], 10, 60, predicate=predicate)
    assert calls and len(calls) == len(set(calls))


def test_post_filtering_would_underfill_where_in_traversal_does_not():
    """Matches live in a cluster far from the query; top-N candidates contain none of them."""
    rng = np.random.default_rng(0)
    near = rng.normal(size=(600, 8)).astype(np.float32) + np.array(
        [8, 0, 0, 0, 0, 0, 0, 0], np.float32
    )
    far = rng.normal(size=(40, 8)).astype(np.float32) - np.array(
        [8, 0, 0, 0, 0, 0, 0, 0], np.float32
    )
    index = HNSWIndex(8, M=12, ef_construction=80, seed=1)
    for i, vector in enumerate(near):
        index.add_item(f"near{i}", vector)
    for i, vector in enumerate(far):
        index.add_item(f"far{i}", vector)
    query = np.array([8, 0, 0, 0, 0, 0, 0, 0], np.float32)
    post = [i for i, _ in index.search(query, 200, 200) if i.startswith("far")]
    filtered = index.search(query, 10, 50, predicate=lambda i: i.startswith("far"))
    assert len(post) < 10
    assert len(filtered) == 10 and all(i.startswith("far") for i, _ in filtered)


def test_filter_after_deletions_never_returns_removed_ids():
    rng = np.random.default_rng(2)
    index = HNSWIndex(6, M=6, ef_construction=40, seed=0)
    for i in range(200):
        index.add_item(str(i), rng.normal(size=6).astype(np.float32))
    for i in range(0, 200, 3):
        index.remove_item(str(i))
    results = index.search(rng.normal(size=6), 30, 80, predicate=lambda i: int(i) % 2 == 0)
    assert results and all(int(i) % 3 != 0 and int(i) % 2 == 0 for i, _ in results)


def test_argument_validation_and_empty_index(dataset):
    index, _ = dataset
    with pytest.raises(ValueError):
        index.search(QUERIES[0], 0, predicate=lambda _: True)
    with pytest.raises(ValueError):
        index.search([1.0, 2.0], 3, predicate=lambda _: True)
    assert HNSWIndex(4).search([1, 0, 0, 0], 3, predicate=lambda _: True) == []


def test_single_node_index_with_filter():
    index = HNSWIndex(2)
    index.add_item("only", [1, 0])
    assert index.search([1, 0], 3, predicate=lambda i: True) == [("only", pytest.approx(1.0))]
    assert index.search([1, 0], 3, predicate=lambda i: False) == []


def test_predicate_exception_propagates_and_leaves_index_usable(dataset):
    index, _ = dataset

    def boom(_):
        raise RuntimeError("bad predicate")

    with pytest.raises(RuntimeError):
        index.search(QUERIES[0], 5, predicate=boom)
    assert len(index.search(QUERIES[0], 5)) == 5
