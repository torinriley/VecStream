"""Native (Rust) engine: correctness, parity with the Python engine, FFI edge cases."""

import ctypes
import gc
import threading

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vecstream import Collection, HNSWIndex
from vecstream.errors import CorruptStoreError, IndexInvariantError, InvalidVectorError
from vecstream.native import NativeHNSWIndex, load_library, native_available, native_status

pytestmark = pytest.mark.skipif(not native_available(), reason="native library not built")

DIM = 24


def data(n=500, dim=DIM, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def truth(vectors, query, k, ids=None):
    ids = np.arange(len(vectors)) if ids is None else np.asarray(ids)
    sub = vectors[ids] / np.maximum(np.linalg.norm(vectors[ids], axis=1, keepdims=True), 1e-12)
    order = np.argsort(-(sub @ (query / np.linalg.norm(query))), kind="stable")[:k]
    return [str(ids[j]) for j in order]


@pytest.fixture(scope="module")
def built():
    vectors = data()
    index = NativeHNSWIndex(DIM, M=12, ef_construction=80, seed=3)
    for i, v in enumerate(vectors):
        index.add_item(str(i), v)
    return index, vectors


def recall(index, vectors, queries, ef=60, k=10):
    hits = 0
    for q in queries:
        want = set(truth(vectors, q, k))
        hits += len(want & {i for i, _ in index.search(q, k, ef)})
    return hits / (k * len(queries))


def test_status_reports_simd_and_library():
    status = native_status()
    assert status["available"] and status["simd"] in ("avx2+fma", "portable")


def test_graph_is_valid_and_recall_is_high(built):
    index, vectors = built
    index.validate()
    assert len(index) == 500
    assert recall(index, vectors, data(40, seed=9)) >= 0.95


def test_scores_match_numpy_cosine(built):
    index, vectors = built
    q = data(1, seed=4)[0]
    for item, score in index.search(q, 10, 100):
        v = vectors[int(item)]
        assert score == pytest.approx(
            float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q))), abs=1e-5
        )


def test_results_are_sorted_best_first_and_unique(built):
    index, _ = built
    rows = index.search(data(1, seed=5)[0], 30, 100)
    scores = [s for _, s in rows]
    assert scores == sorted(scores, reverse=True) and len({i for i, _ in rows}) == len(rows)


def test_parity_with_python_engine(built):
    index, vectors = built
    reference = HNSWIndex(DIM, M=12, ef_construction=80, seed=3)
    for i, v in enumerate(vectors):
        reference.add_item(str(i), v)
    queries = data(40, seed=11)
    native_recall = recall(index, vectors, queries)
    python_recall = recall(reference, vectors, queries)
    assert native_recall >= python_recall - 0.03


def test_simd_and_portable_kernels_return_the_same_neighbors(built):
    index, _ = built
    lib = load_library()
    q = data(1, seed=6)[0]
    simd = index.search(q, 10, 80)
    lib.vs_force_scalar(1)
    try:
        portable = index.search(q, 10, 80)
    finally:
        lib.vs_force_scalar(0)
    assert [i for i, _ in simd][:5] == [i for i, _ in portable][:5]
    assert [s for _, s in simd] == pytest.approx([s for _, s in portable], abs=1e-5)


@pytest.mark.parametrize("dim", [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65, 100])
def test_odd_dimensions_work(dim):
    vectors = data(60, dim, seed=dim)
    index = NativeHNSWIndex(dim, M=4, ef_construction=16, seed=1)
    for i, v in enumerate(vectors):
        index.add_item(str(i), v)
    index.validate()
    top = index.search(vectors[7], 1, 30)
    assert top and top[0][1] == pytest.approx(1.0, abs=1e-5)
    if dim >= 3:  # in 1-2 dimensions many vectors are exact direction ties
        assert top[0][0] == "7"


def test_updates_removals_and_slot_reuse():
    index = NativeHNSWIndex(4, M=4, ef_construction=16, seed=2)
    v = data(30, 4, seed=1)
    for i, x in enumerate(v):
        index.add_item(f"n{i}", x)
    index.add_item("n3", v[0])  # update in place
    assert len(index) == 30
    for i in range(0, 30, 2):
        index.remove_item(f"n{i}")
        index.validate()
    with pytest.raises(KeyError):
        index.remove_item("n0")
    assert "n0" not in index.nodes and "n1" in index.nodes
    for i in range(10):
        index.add_item(f"new{i}", v[i])  # reuses freed slots
    index.validate()
    assert len(index) == 25
    assert {i for i, _ in index.search(v[1], 25, 60)} == set(index.nodes)


def test_empty_index_and_remove_all():
    index = NativeHNSWIndex(3)
    assert index.search([1, 0, 0], 5) == [] and index.search_batch([[1, 0, 0]], 3) == [[]]
    index.add_item("a", [1, 0, 0])
    index.remove_item("a")
    assert index.search([1, 0, 0], 5) == []
    index.validate()
    index.add_item("b", [0, 1, 0])
    assert index.search([0, 1, 0], 1)[0][0] == "b"


def test_k_larger_than_index_returns_everything():
    index = NativeHNSWIndex(3, M=4, ef_construction=8, seed=1)
    for i in range(5):
        index.add_item(str(i), data(5, 3)[i])
    assert len(index.search([1, 1, 1], 100, 100)) == 5


def test_input_validation_happens_before_the_ffi_boundary():
    index = NativeHNSWIndex(3)
    with pytest.raises(InvalidVectorError):
        index.add_item("a", [1, 2])
    with pytest.raises(InvalidVectorError):
        index.add_item("a", [1, float("nan"), 0])
    with pytest.raises(InvalidVectorError):
        index.search([1, 2], 1)
    with pytest.raises(ValueError):
        index.search([1, 2, 3], 0)
    with pytest.raises(ValueError):
        NativeHNSWIndex(0)
    with pytest.raises(ValueError):
        NativeHNSWIndex(4, M=1)
    assert index.nodes == {}


def test_zero_vector_is_stored_and_never_matches_with_positive_score():
    index = NativeHNSWIndex(3, M=4, ef_construction=8, seed=1)
    index.add_item("zero", [0, 0, 0])
    index.add_item("x", [1, 0, 0])
    scores = dict(index.search([1, 0, 0], 2, 10))
    assert scores["x"] == pytest.approx(1.0) and scores["zero"] == pytest.approx(0.0)
    index.validate()


def test_predicate_min_score_and_fallback(built):
    index, vectors = built
    q = data(1, seed=8)[0]
    even = lambda i: int(i) % 2 == 0  # noqa: E731
    rows = index.search(q, 10, 60, predicate=even)
    assert len(rows) == 10 and all(int(i) % 2 == 0 for i, _ in rows)
    assert {i for i, _ in rows} & set(truth(vectors, q, 10, np.arange(0, 500, 2)))
    rare = {"3", "77", "401"}
    got = [i for i, _ in index.search(q, 3, 8, predicate=rare.__contains__, max_visits=30)]
    assert got == [i for i in truth(vectors, q, 500) if i in rare][:3]
    scored = index.search(q, 100, 200, min_score=0.25)
    assert scored and all(s >= 0.25 - 1e-5 for _, s in scored)
    assert index.search(q, 5, 20, min_score=1.5) == []
    assert index.search(q, 5, 20, predicate=lambda _: False) == []


def test_predicate_exception_propagates_and_index_stays_usable(built):
    index, _ = built

    class Boom(Exception):
        pass

    def bad(_):
        raise Boom

    with pytest.raises(Boom):
        index.search(data(1)[0], 5, predicate=bad)
    assert len(index.search(data(1)[0], 5)) == 5


def test_batch_matches_sequential_for_any_thread_count(built):
    index, _ = built
    queries = data(37, seed=12)
    sequential = [index.search(q, 7, 50) for q in queries]
    for threads in (1, 2, 4, 64):
        batch = index.search_batch(queries, 7, 50, threads=threads)
        assert [[i for i, _ in r] for r in batch] == [[i for i, _ in r] for r in sequential]
    assert index.search_batch(queries[:1], 7, 50, min_score=2.0) == [[]]


def test_concurrent_python_threads_searching_one_index(built):
    index, vectors = built
    queries = data(40, seed=13)
    expected = [index.search(q, 5, 40) for q in queries]
    errors = []

    def worker():
        try:
            for _ in range(3):
                for q, want in zip(queries, expected, strict=True):
                    assert index.search(q, 5, 40) == want
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_topology_round_trip_between_engines():
    vectors = data(150, 12, seed=2)
    ids = [f"id{i}" for i in range(150)]
    source = HNSWIndex(12, M=6, ef_construction=40, seed=1)
    for item, v in zip(ids, vectors, strict=True):
        source.add_item(item, v)
    for item in ids[::7]:
        source.remove_item(item)
    order = [i for i in ids if i in source.nodes]
    levels, edges, ep = source.export_topology(order)
    raw = {i: source.nodes[i] for i in order}
    native = NativeHNSWIndex.from_topology(12, 6, 40, None, order, raw, levels, edges, ep)
    native.validate()
    for q in data(10, 12, seed=3):
        assert [i for i, _ in native.search(q, 10, 50)] == [i for i, _ in source.search(q, 10, 50)]
    levels2, edges2, ep2 = native.export_topology(order)
    assert np.array_equal(levels, levels2) and ep == ep2
    assert {tuple(e) for e in edges.tolist()} == {tuple(e) for e in edges2.tolist()}
    back = HNSWIndex.from_topology(12, 6, 40, None, order, raw, levels2, edges2, ep2)
    back.validate()


def test_corrupt_topology_is_rejected_not_trusted():
    vectors = data(30, 6, seed=3)
    ids = [str(i) for i in range(30)]
    index = NativeHNSWIndex(6, M=4, ef_construction=16, seed=1)
    for item, v in zip(ids, vectors, strict=True):
        index.add_item(item, v)
    levels, edges, ep = index.export_topology(ids)
    raw = dict(zip(ids, vectors, strict=True))
    for broken in (edges[:-1], np.vstack([edges, [0, 0, 29]]), np.vstack([edges, [0, 0, 500]])):
        with pytest.raises(CorruptStoreError):
            NativeHNSWIndex.from_topology(6, 4, 16, None, ids, raw, levels, broken, ep)
    with pytest.raises(CorruptStoreError):
        NativeHNSWIndex.from_topology(6, 4, 16, None, ids, raw, levels, edges, 999)


def test_export_requires_matching_ids():
    index = NativeHNSWIndex(3)
    index.add_item("a", [1, 0, 0])
    with pytest.raises(IndexInvariantError):
        index.export_topology(["a", "b"])


@settings(max_examples=20, deadline=None)
@given(
    ops=st.lists(
        st.tuples(st.sampled_from(["add", "remove"]), st.integers(0, 24)), min_size=1, max_size=70
    )
)
def test_random_mutations_keep_native_invariants(ops):
    pool = data(25, 5, seed=1)
    index = NativeHNSWIndex(5, M=3, ef_construction=8, seed=2)
    live = set()
    for kind, n in ops:
        if kind == "add":
            index.add_item(str(n), pool[n])
            live.add(str(n))
        elif str(n) in live:
            index.remove_item(str(n))
            live.discard(str(n))
        index.validate()
        assert set(index.nodes) == live
        if live:
            assert index.search(pool[n], 1, 8)


def test_deletion_repair_recall():
    vectors = data(600, DIM, seed=5)
    index = NativeHNSWIndex(DIM, M=12, ef_construction=80, seed=1)
    for i, v in enumerate(vectors):
        index.add_item(str(i), v)
    doomed = np.random.default_rng(3).permutation(600)[:240]
    for i in doomed:
        index.remove_item(str(int(i)))
    index.validate()
    live = np.setdiff1d(np.arange(600), doomed)
    hits = 0
    queries = data(40, seed=21)
    for q in queries:
        hits += len(set(truth(vectors, q, 10, live)) & {i for i, _ in index.search(q, 10, 60)})
    assert hits / 400 >= 0.9


def test_creating_and_dropping_many_indexes_does_not_grow_memory():
    def rss():
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * 4096

    vectors = data(200, 16, seed=1)

    def cycle():
        index = NativeHNSWIndex(16, M=8, ef_construction=32, seed=1)
        for i, v in enumerate(vectors):
            index.add_item(str(i), v)
        del index
        gc.collect()

    for _ in range(3):
        cycle()
    before = rss()
    for _ in range(25):
        cycle()
    assert rss() - before < 60 * 1024 * 1024


def test_abi_version_matches_the_python_bindings():
    from vecstream.native import ABI_VERSION

    assert load_library().vs_abi_version() == ABI_VERSION == 2


def test_implausible_parameters_are_rejected_instead_of_aborting_the_process():
    lib = load_library()
    for dim, m, ef in [
        (4, 4_000_000_000, 4_000_000_000),
        (4, 1025, 2000),
        (4, 16, 2_000_000),
        (70_000, 16, 100),
    ]:
        assert not lib.vs_new(dim, m, ef, -1, 1), (dim, m, ef)
    for kwargs in ({"M": 10**9, "ef_construction": 10**9}, {"M": 16, "ef_construction": 10**7}):
        with pytest.raises(ValueError):
            NativeHNSWIndex(8, **kwargs)
    from vecstream.collections import validate_hnsw_params

    for bad in ({"M": 10**9}, {"M": 16, "ef_construction": 10**7}):
        with pytest.raises(ValueError):
            validate_hnsw_params(bad)


def test_import_rejects_absurd_levels_without_allocating_them():
    lib = load_library()
    vectors = (ctypes.c_float * 2)(1, 0)
    for level in (65, 2_000_000_000):
        levels = (ctypes.c_int32 * 1)(level)
        assert not lib.vs_import(2, 4, 16, -1, 1, 1, vectors, levels, 0, None, 0)


def test_export_refuses_undersized_buffers_and_never_writes_past_them():
    lib = load_library()
    handle = lib.vs_new(2, 4, 16, -1, 1)
    try:
        for i in range(20):
            assert lib.vs_add(handle, (ctypes.c_float * 2)(1.0, float(i))) >= 0
        sizes = (ctypes.c_int64 * 3)()
        assert lib.vs_export_sizes(handle, sizes) == 0
        slots, edge_count = int(sizes[0]), int(sizes[1])
        lib.vs_add(handle, (ctypes.c_float * 2)(5.0, 5.0))  # index grows after sizing
        sentinel = -7
        levels = (ctypes.c_int32 * (slots + 4))(*([sentinel] * (slots + 4)))
        roomy = (edge_count + 40) * 3  # large enough for the final call's declared capacity
        edges = (ctypes.c_int32 * roomy)(*([sentinel] * roomy))
        assert lib.vs_export(handle, levels, slots, edges, edge_count) == -4  # ERR_CAPACITY
        assert set(levels) == {sentinel} and set(edges) == {sentinel}  # nothing was written
        assert lib.vs_export(handle, levels, slots + 4, edges, edge_count + 40) == 0
    finally:
        lib.vs_free(handle)


def test_raw_handle_is_safe_under_unsynchronized_threads():
    """Mutations and searches from many threads must block, not race (no crash, valid graph)."""
    index = NativeHNSWIndex(8, M=6, ef_construction=24, seed=1)
    pool = data(300, 8, seed=2)
    for i in range(50):
        index.add_item(f"s{i}", pool[i])
    stop = threading.Event()
    errors = []

    def searcher():
        try:
            while not stop.is_set():
                index.search(pool[0], 5, 20)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=searcher) for _ in range(3)]
    for t in threads:
        t.start()
    try:
        for i in range(50, 250):
            index.add_item(f"s{i}", pool[i])
            if i % 3 == 0:
                index.remove_item(f"s{i - 40}")
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert errors == []
    index.validate()


def test_raw_ffi_rejects_bad_arguments():
    lib = load_library()
    assert not lib.vs_new(0, 16, 100, -1, 1)
    assert not lib.vs_new(4, 1, 100, -1, 1)
    assert lib.vs_len(None) < 0
    assert lib.vs_remove(None, 0) < 0
    handle = lib.vs_new(3, 4, 16, -1, 1)
    try:
        assert lib.vs_remove(handle, 12345) == 0  # unknown slot is a no-op, not a crash
        nan = (ctypes.c_float * 3)(1.0, float("nan"), 0.0)
        assert lib.vs_add(handle, nan) < 0
        assert lib.vs_add(handle, None) < 0
        ok = (ctypes.c_float * 3)(1.0, 0.0, 0.0)
        assert lib.vs_add(handle, ok) == 0
        slots, scores = (ctypes.c_uint32 * 2)(), (ctypes.c_float * 2)()
        assert (
            lib.vs_search(
                handle,
                ok,
                0,
                8,
                float("nan"),
                ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint32)(0),
                0,
                slots,
                scores,
                None,
            )
            < 0
        )
    finally:
        lib.vs_free(handle)
    lib.vs_free(None)  # freeing null is allowed


# ---- Collection integration -------------------------------------------------------------


def make_collection(path, engine, count=200, **kw):
    c = Collection(
        "c", str(path), True, {"M": 8, "ef_construction": 48}, sync=False, engine=engine, **kw
    )
    c.apply(
        [
            {"op": "put", "id": f"v{i}", "vector": v, "metadata": {"g": i % 5}}
            for i, v in enumerate(data(count, 12, seed=7))
        ]
    )
    return c


@pytest.mark.parametrize(
    "writer,reader", [("python", "native"), ("native", "python"), ("native", "native")]
)
def test_checkpoints_load_across_engines(tmp_path, writer, reader):
    first = make_collection(tmp_path, writer)
    queries = data(10, 12, seed=8)
    want = [first.search_similar(q, 10, ef_search=60) for q in queries]
    first.close()
    second = Collection(
        "c", str(tmp_path), True, {"M": 8, "ef_construction": 48}, sync=False, engine=reader
    )
    assert second.engine == reader and second.recovery["graph_restored"] is True
    second.verify()
    for q, expected in zip(queries, want, strict=True):
        got = second.search_similar(q, 10, ef_search=60)
        assert len({i for i, _ in got} & {i for i, _ in expected}) >= 9
    second.close()


def test_native_collection_replays_log_filters_and_batches(tmp_path):
    c = make_collection(tmp_path, "native")
    c.checkpoint()
    c.add_vector("late", data(1, 12, seed=1)[0], {"g": 99})
    c.remove_vector("v3")
    import shutil

    shutil.copytree(tmp_path, tmp_path.parent / (tmp_path.name + "-crash"))
    crashed = tmp_path.parent / (tmp_path.name + "-crash")
    (crashed / "collections" / "c" / "LOCK").unlink()
    r = Collection(
        "c", str(crashed), True, {"M": 8, "ef_construction": 48}, sync=False, engine="native"
    )
    assert r.recovery["replayed_records"] == 2 and "v3" not in r.store.vectors
    r.verify()
    assert [
        i for i, _ in r.search_similar(data(1, 12, seed=1)[0], 1, filter_metadata={"g": 99})
    ] == ["late"]
    queries = data(9, 12, seed=2)
    rows = r.search_batch(queries, k=5, ef_search=40)
    assert len(rows) == 9 and all(len(row) == 5 for row in rows)
    assert r.health()["engine"] == "native"


def test_corrupt_persisted_graph_falls_back_to_rebuild_on_native(tmp_path):
    import hashlib
    import json

    c = make_collection(tmp_path, "native", count=60)
    c.close()
    root = tmp_path / "collections" / "c"
    current = json.loads((root / "CURRENT").read_text())["current"]
    manifest = json.loads((root / current / "manifest.json").read_text())
    tampered = (root / current / "hnsw.json").read_bytes().replace(b'"nodes": 60', b'"nodes": 61')
    (root / current / "hnsw.json").write_bytes(tampered)
    manifest["checksums"]["hnsw.json"] = hashlib.sha256(tampered).hexdigest()
    (root / current / "manifest.json").write_text(json.dumps(manifest))
    r = Collection(
        "c", str(tmp_path), True, {"M": 8, "ef_construction": 48}, sync=False, engine="native"
    )
    assert r.recovery["graph_restored"] is False and len(r.hnsw_index.nodes) == 60
    r.verify()


def test_engine_selection_rules(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        Collection("x", str(tmp_path), engine="gpu")
    assert Collection("a", str(tmp_path), engine="python").engine == "python"
    assert Collection("b", str(tmp_path), engine="auto").engine == "native"
    monkeypatch.setattr("vecstream.collections.native_available", lambda: False)
    assert Collection("c", str(tmp_path), engine="auto").engine == "python"
    with pytest.raises(RuntimeError):
        Collection("d", str(tmp_path), engine="native")
