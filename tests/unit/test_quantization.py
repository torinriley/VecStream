"""8-bit scalar quantization: memory, accuracy, persistence, and the exact re-ranking step."""

import json

import numpy as np
import pytest

from vecstream import Collection, CollectionManager
from vecstream.collections import RESCORE_FACTOR, validate_hnsw_params
from vecstream.hnsw_index import HNSWIndex
from vecstream.native import NativeHNSWIndex, native_available

native_only = pytest.mark.skipif(not native_available(), reason="native library not built")

DIM = 48


def data(n, dim=DIM, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def exact_top(vectors, query, k):
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    scores = unit @ (query / np.linalg.norm(query))
    order = np.argsort(-scores, kind="stable")[:k]
    return [(f"v{i}", float(scores[i])) for i in order]


def fill(collection, vectors, metadata=None):
    collection.apply(
        [
            {
                "op": "put",
                "id": f"v{i}",
                "vector": v,
                "metadata": metadata(i) if metadata else None,
            }
            for i, v in enumerate(vectors)
        ]
    )


def make(path, quantization, vectors, **params):
    c = Collection(
        "c",
        str(path),
        True,
        {"M": 12, "ef_construction": 80, "quantization": quantization, **params},
        sync=False,
        engine="native",
    )
    fill(c, vectors, lambda i: {"g": i % 4})
    return c


def test_parameter_validation_does_not_need_the_native_library():
    for ok in ("none", "sq8"):
        assert validate_hnsw_params({"quantization": ok})["quantization"] == ok
    for bad in ("sq4", "fp16", "", 8, None, True):
        with pytest.raises(ValueError):
            validate_hnsw_params({"quantization": bad})


def test_python_engine_refuses_quantization_instead_of_ignoring_it(tmp_path):
    with pytest.raises(ValueError, match="native engine"):
        Collection("c", str(tmp_path), True, {"quantization": "sq8"}, engine="python")
    with pytest.raises(ValueError):
        HNSWIndex.from_topology(
            4, 4, 8, None, [], {}, np.zeros(0, np.int32), np.zeros((0, 3), np.int32), None, "sq8"
        )


@native_only
def test_index_memory_shrinks_to_about_a_quarter():
    vectors = data(400, 128)
    plain = NativeHNSWIndex(128, M=8, ef_construction=40, seed=1)
    packed = NativeHNSWIndex(128, M=8, ef_construction=40, seed=1, quantization="sq8")
    for i, v in enumerate(vectors):
        plain.add_item(str(i), v)
        packed.add_item(str(i), v)
    assert plain.stats()["vector_bytes"] == 400 * 128 * 4
    assert packed.stats()["vector_bytes"] == 400 * (128 + 12)
    assert packed.stats()["vector_bytes"] < 0.3 * plain.stats()["vector_bytes"]
    packed.validate()


@native_only
def test_unknown_mode_is_rejected_by_the_wrapper():
    with pytest.raises(ValueError):
        NativeHNSWIndex(8, quantization="int4")


@native_only
def test_raw_quantized_scores_are_close_but_collection_scores_are_exact(tmp_path):
    vectors = data(300)
    query = data(1, seed=5)[0]
    raw = NativeHNSWIndex(DIM, M=12, ef_construction=80, seed=1, quantization="sq8")
    for i, v in enumerate(vectors):
        raw.add_item(f"v{i}", v)
    truth = dict(exact_top(vectors, query, 300))
    approx = raw.search(query, 10, 100)
    errors = [abs(score - truth[item]) for item, score in approx]
    assert max(errors) < 0.02 and max(errors) > 0  # approximate, but close
    collection = make(tmp_path, "sq8", vectors)
    for item, score in collection.search_similar(query, 10, ef_search=100):
        assert score == pytest.approx(truth[item], abs=1e-6)  # re-ranked from stored floats


@native_only
def test_collection_recall_stays_close_to_unquantized(tmp_path):
    vectors = data(800)
    plain = make(tmp_path / "plain", "none", vectors)
    packed = make(tmp_path / "packed", "sq8", vectors)
    queries = data(40, seed=9)

    def recall(collection):
        hits = 0
        for q in queries:
            want = {i for i, _ in exact_top(vectors, q, 10)}
            hits += len(want & {i for i, _ in collection.search_similar(q, 10, ef_search=60)})
        return hits / (10 * len(queries))

    assert recall(packed) >= recall(plain) - 0.04 and recall(packed) >= 0.9


@native_only
def test_results_are_sorted_unique_and_respect_k_and_threshold(tmp_path):
    vectors = data(300)
    c = make(tmp_path, "sq8", vectors)
    q = data(1, seed=6)[0]
    rows = c.search_similar(q, 12, ef_search=80)
    scores = [s for _, s in rows]
    assert len(rows) == 12 and scores == sorted(scores, reverse=True)
    assert len({i for i, _ in rows}) == 12
    high = c.search_similar(q, 50, threshold=0.2, ef_search=200)
    assert high and all(s >= 0.2 for _, s in high)
    assert c.search_similar(q, 5, threshold=1.01) == []


@native_only
def test_filters_and_batches_work_with_quantization(tmp_path):
    vectors = data(400)
    c = make(tmp_path, "sq8", vectors)
    q = data(1, seed=7)[0]
    rows = c.search_similar(q, 10, filter_metadata={"g": 2}, ef_search=80)
    assert len(rows) == 10 and all(int(i[1:]) % 4 == 2 for i, _ in rows)
    queries = data(9, seed=8)
    batch = c.search_batch(queries, k=5, ef_search=60)
    assert len(batch) == 9 and all(len(row) == 5 for row in batch)
    for query, row in zip(queries, batch, strict=True):
        truth = dict(exact_top(vectors, query, 400))
        for item, score in row:
            assert score == pytest.approx(truth[item], abs=1e-6)  # exact, not approximate
    assert c.search_batch(queries, k=5, threshold=0.99) == [[] for _ in queries]


@native_only
def test_updates_deletes_and_vectors_without_variance(tmp_path):
    vectors = data(120)
    c = make(tmp_path, "sq8", vectors)
    c.apply([{"op": "del", "id": f"v{i}"} for i in range(0, 120, 3)])
    c.add_vector("flat", [0.5] * DIM)  # constant vector: scale 0
    c.add_vector("zero", [0.0] * DIM)
    c.verify()
    top = c.search_similar([0.5] * DIM, 1, ef_search=50)
    assert top[0][0] == "flat" and top[0][1] == pytest.approx(1.0, abs=1e-6)
    everything = dict(c.search_similar([0.5] * DIM, 150, ef_search=200))
    assert everything["zero"] == 0.0  # cosine with the zero vector is defined as 0
    positive = dict(c.search_similar([0.5] * DIM, 150, threshold=0.01, ef_search=200))
    assert "zero" not in positive
    deleted = {f"v{i}" for i in range(0, 120, 3)}
    assert deleted.isdisjoint(i for i, _ in c.search_similar(vectors[0], 100, ef_search=200))


@native_only
def test_checkpoint_round_trip_and_log_replay_keep_quantization(tmp_path):
    vectors = data(200)
    c = make(tmp_path, "sq8", vectors)
    q = data(1, seed=3)[0]
    before = c.search_similar(q, 10, ef_search=80)
    c.close()
    again = Collection(
        "c",
        str(tmp_path),
        True,
        {"M": 12, "ef_construction": 80, "quantization": "sq8"},
        sync=False,
        engine="native",
    )
    assert again.recovery["graph_restored"] is True and again.quantization == "sq8"
    assert again.hnsw_index.quantization == "sq8"
    assert again.search_similar(q, 10, ef_search=80) == before
    assert again.health()["quantization"] == "sq8"
    assert again.health()["index_vector_bytes"] == 200 * (DIM + 12)


@native_only
def test_changing_quantization_reuses_the_saved_graph(tmp_path):
    """Topology does not depend on how vectors are stored, so switching modes needs no rebuild."""
    vectors = data(200)
    c = make(tmp_path, "none", vectors)
    c.close()
    packed = Collection(
        "c",
        str(tmp_path),
        True,
        {"M": 12, "ef_construction": 80, "quantization": "sq8"},
        sync=False,
        engine="native",
    )
    assert packed.recovery["graph_restored"] is True
    assert packed.hnsw_index.quantization == "sq8"
    packed.verify()
    assert packed.health()["index_vector_bytes"] == 200 * (DIM + 12)
    packed.close()
    unpacked = Collection(
        "c",
        str(tmp_path),
        True,
        {"M": 12, "ef_construction": 80, "quantization": "none"},
        sync=False,
        engine="native",
    )
    assert unpacked.recovery["graph_restored"] is True
    assert unpacked.hnsw_index.quantization == "none"


@native_only
def test_manager_saves_quantization_and_applies_it_on_reopen(tmp_path):
    manager = CollectionManager(str(tmp_path), sync=False, engine="native")
    created = manager.create_collection("q", hnsw_params={"quantization": "sq8"})
    assert created.hnsw_params == {"M": 32, "ef_construction": 200, "quantization": "sq8"}
    fill(created, data(60))
    manager.close()
    saved = json.loads((tmp_path / "collections_metadata.json").read_text())
    assert saved["q"]["hnsw_params"]["quantization"] == "sq8"
    reopened = CollectionManager(str(tmp_path), sync=False, engine="native").get_collection("q")
    assert reopened.quantization == "sq8" and reopened.recovery["graph_restored"] is True
    plain = CollectionManager(str(tmp_path / "p"), sync=False, engine="native")
    assert plain.create_collection("x").quantization == "none"


@native_only
def test_oversampling_gives_the_reranker_enough_candidates(tmp_path):
    vectors = data(200)
    c = make(tmp_path, "sq8", vectors)
    seen = []
    original = c.hnsw_index.search

    def spy(query, k=10, **kwargs):
        seen.append((k, kwargs.get("ef_search")))
        return original(query, k=k, **kwargs)

    c.hnsw_index.search = spy
    c.search_similar(data(1, seed=2)[0], 5, ef_search=10)
    assert seen == [(5 * RESCORE_FACTOR, 5 * RESCORE_FACTOR)]  # ef raised to cover the oversample
