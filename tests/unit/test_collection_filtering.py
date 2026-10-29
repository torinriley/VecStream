"""Collection-level metadata filtering and thresholds with the in-traversal search."""

import numpy as np
import pytest

from vecstream import Collection


def make(tmp_path, count=300, use_hnsw=True, **options):
    collection = Collection(
        "c", str(tmp_path), use_hnsw, {"M": 10, "ef_construction": 60}, sync=False, **options
    )
    rng = np.random.default_rng(4)
    vectors = rng.normal(size=(count, 12)).astype(np.float32)
    collection.apply(
        [
            {
                "op": "put",
                "id": f"v{i}",
                "vector": vector,
                "metadata": {"group": i % 10, "even": i % 2 == 0, "nest": {"tier": i % 3}},
            }
            for i, vector in enumerate(vectors)
        ]
    )
    return collection, vectors


def test_filter_returns_k_results_all_matching(tmp_path):
    collection, vectors = make(tmp_path)
    results = collection.search_similar(vectors[0], k=10, filter_metadata={"group": 7})
    assert len(results) == 10
    assert all(int(item[1:]) % 10 == 7 for item, _ in results)


def test_filter_fills_k_when_matches_are_rare(tmp_path):
    collection, vectors = make(tmp_path)
    collection.add_vector("rare", -vectors[0], {"group": 99})
    collection.add_vector("rare2", -vectors[1], {"group": 99})
    results = collection.search_similar(
        vectors[0], k=5, threshold=-1.0, filter_metadata={"group": 99}
    )
    assert {item for item, _ in results} == {"rare", "rare2"}


def test_dotted_key_and_multi_key_filters(tmp_path):
    collection, vectors = make(tmp_path)
    results = collection.search_similar(
        vectors[3], k=8, filter_metadata={"nest.tier": 1, "even": True}
    )
    assert results
    for item, _ in results:
        i = int(item[1:])
        assert i % 3 == 1 and i % 2 == 0


def test_filter_matching_nothing_and_empty_filter(tmp_path):
    collection, vectors = make(tmp_path)
    assert collection.search_similar(vectors[0], k=5, filter_metadata={"group": 12345}) == []
    assert collection.search_similar(
        vectors[0], k=5, filter_metadata={}
    ) == collection.search_similar(vectors[0], k=5)


def test_vectors_without_metadata_never_match_a_filter(tmp_path):
    collection, vectors = make(tmp_path, count=50)
    collection.add_vector("bare", vectors[0])
    results = collection.search_similar(vectors[0], k=60, filter_metadata={"group": 0})
    assert "bare" not in {item for item, _ in results}


def test_threshold_is_enforced_during_search_and_fills_k(tmp_path):
    collection, vectors = make(tmp_path)
    results = collection.search_similar(vectors[0], k=20, threshold=0.2, ef_search=100)
    assert results and all(score >= 0.2 - 1e-6 for _, score in results)
    unconstrained = collection.search_similar(vectors[0], k=300, ef_search=300, threshold=0.2)
    assert len(unconstrained) >= len(results)


def test_default_threshold_still_excludes_negative_similarity(tmp_path):
    collection, vectors = make(tmp_path)
    collection.add_vector("opposite", -vectors[0], {"group": 1})
    ids = {item for item, _ in collection.search_similar(vectors[0], k=300, ef_search=300)}
    assert "opposite" not in ids
    ids = {
        item
        for item, _ in collection.search_similar(
            vectors[0], k=300, ef_search=300, threshold=-1.0, filter_metadata={"group": 1}
        )
    }
    assert "opposite" in ids


def test_filter_with_threshold_combines(tmp_path):
    collection, vectors = make(tmp_path)
    results = collection.search_similar(
        vectors[0], k=10, threshold=0.1, filter_metadata={"even": True}
    )
    assert results
    assert all(score >= 0.1 - 1e-6 and int(item[1:]) % 2 == 0 for item, score in results)


def test_hnsw_filter_agrees_with_exact_path(tmp_path):
    hnsw, vectors = make(tmp_path / "h", count=250)
    exact, _ = make(tmp_path / "e", count=250, use_hnsw=False)
    hits = total = 0
    for query in vectors[:25]:
        want = {i for i, _ in exact.search_similar(query, k=10, filter_metadata={"group": 4})}
        got = {i for i, _ in hnsw.search_similar(query, k=10, filter_metadata={"group": 4})}
        hits, total = hits + len(want & got), total + len(want)
    assert hits / total >= 0.9


def test_updates_and_deletes_are_reflected_in_filtered_results(tmp_path):
    collection, vectors = make(tmp_path, count=100)
    collection.add_vector("v0", vectors[0], {"group": 555})
    assert [
        i for i, _ in collection.search_similar(vectors[0], 3, filter_metadata={"group": 555})
    ] == ["v0"]
    collection.remove_vector("v0")
    assert collection.search_similar(vectors[0], 3, filter_metadata={"group": 555}) == []
    collection.verify()


def test_filtered_search_works_on_a_graph_restored_from_disk(tmp_path):
    collection, vectors = make(tmp_path)
    expected = collection.search_similar(vectors[0], k=10, filter_metadata={"group": 2})
    collection.close()
    reopened = Collection("c", str(tmp_path), True, {"M": 10, "ef_construction": 60}, sync=False)
    assert reopened.recovery["graph_restored"] is True
    assert reopened.search_similar(vectors[0], k=10, filter_metadata={"group": 2}) == expected


def test_search_detailed_returns_metadata_under_same_lock(tmp_path):
    collection, vectors = make(tmp_path, count=60)
    rows = collection.search_detailed(vectors[0], k=3, filter_metadata={"even": True})
    assert rows and all(metadata["even"] is True for _, _, metadata in rows)


def test_deleting_most_vectors_keeps_search_correct_and_connected(tmp_path):
    collection, vectors = make(tmp_path, count=200)
    collection.apply([{"op": "del", "id": f"v{i}"} for i in range(200) if i % 5 != 0])
    collection.verify()
    results = collection.search_similar(vectors[10], k=5, ef_search=50)
    assert len(results) == 5 and all(int(i[1:]) % 5 == 0 for i, _ in results)


@pytest.mark.parametrize("bad_filter", [{"missing": 1}, {"nest.nope": 1}, {"group.sub": 1}])
def test_filters_on_absent_or_non_object_paths_match_nothing(tmp_path, bad_filter):
    collection, vectors = make(tmp_path, count=60)
    assert collection.search_similar(vectors[0], k=5, filter_metadata=bad_filter) == []
