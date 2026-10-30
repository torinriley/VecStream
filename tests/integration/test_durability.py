import json
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import threading

import numpy as np
import pytest

from vecstream import Collection, CollectionManager, HNSWIndex
from vecstream.errors import ClosedError, CorruptStoreError, StoreLockedError


def _vectors(count, dim=8, seed=0):
    return np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32)


def _open(path, **kwargs):
    kwargs.setdefault("sync", False)
    return Collection("c", str(path), True, {"M": 8, "ef_construction": 40}, **kwargs)


def test_graph_is_restored_without_rebuilding(tmp_path, monkeypatch):
    collection = _open(tmp_path)
    for i, vector in enumerate(_vectors(120)):
        collection.add_vector(f"v{i}", vector, {"i": i})
    expected = collection.search_similar(_vectors(1, seed=9)[0], k=5)
    collection.close()

    inserts = []
    original = HNSWIndex.add_item
    monkeypatch.setattr(
        HNSWIndex, "add_item", lambda self, *a, **k: inserts.append(a) or original(self, *a, **k)
    )
    reopened = _open(tmp_path)
    assert reopened.recovery["graph_restored"] is True
    assert reopened.recovery["replayed_records"] == 0
    assert inserts == []
    assert reopened.search_similar(_vectors(1, seed=9)[0], k=5) == expected
    reopened.verify()


def test_unclean_shutdown_replays_log_into_restored_graph(tmp_path):
    live = tmp_path / "live"
    collection = _open(live)
    for i, vector in enumerate(_vectors(50)):
        collection.add_vector(f"v{i}", vector)
    collection.checkpoint()
    collection.add_vector("late", _vectors(1, seed=5)[0], {"after": "checkpoint"})
    collection.remove_vector("v3")
    crashed = tmp_path / "crashed"
    shutil.copytree(live, crashed)  # the files a SIGKILL would leave behind
    recovered = _open(crashed)
    assert recovered.recovery["graph_restored"] and recovered.recovery["replayed_records"] == 2
    assert recovered.get_vector_with_metadata("late")[1] == {"after": "checkpoint"}
    with pytest.raises(KeyError):
        recovered.get_vector_with_metadata("v3")
    assert "v3" not in recovered.hnsw_index.nodes
    recovered.verify()


def test_batch_is_all_or_nothing(tmp_path):
    collection = _open(tmp_path)
    collection.add_vector("keep", [1.0] * 8)
    lsn_before = collection.wal.last_lsn
    bad_batches = [
        [{"op": "put", "id": "x", "vector": [1.0] * 8}, {"op": "del", "id": "missing"}],
        [{"op": "put", "id": "x", "vector": [1.0] * 8}, {"op": "put", "id": "y", "vector": [1.0]}],
        [
            {"op": "put", "id": "x", "vector": [1.0] * 8},
            {"op": "put", "id": "z", "vector": [float("nan")] * 8},
        ],
        [{"op": "put", "id": "x", "vector": [1.0] * 8, "metadata": {"bad": float("inf")}}],
    ]
    for batch in bad_batches:
        with pytest.raises((ValueError, KeyError)):
            collection.apply(batch)
    assert collection.wal.last_lsn == lsn_before
    assert set(collection.store.vectors) == {"keep"}
    collection.apply(
        [
            {"op": "put", "id": "a", "vector": [0.5] * 8},
            {"op": "del", "id": "keep"},
            {"op": "put", "id": "b", "vector": [0.25] * 8, "metadata": {"k": 1}},
        ]
    )
    assert set(collection.store.vectors) == {"a", "b"}


def test_failed_log_append_leaves_state_untouched(tmp_path, monkeypatch):
    collection = _open(tmp_path)
    collection.add_vector("a", [1.0] * 8)

    def boom(payload):
        raise OSError("disk full")

    monkeypatch.setattr(collection.wal, "append", boom)
    with pytest.raises(OSError):
        collection.add_vector("b", [2.0] * 8)
    assert list(collection.store.vectors) == ["a"]
    assert list(collection.hnsw_index.nodes) == ["a"]


def test_crash_between_checkpoint_and_log_reset_does_not_double_apply(tmp_path, monkeypatch):
    collection = _open(tmp_path)
    collection.add_vector("a", [1.0] * 8, {"n": 1})
    collection.remove_vector("a")
    collection.add_vector("a", [2.0] * 8, {"n": 2})
    monkeypatch.setattr(
        collection.wal, "reset", lambda lsn: (_ for _ in ()).throw(OSError("crash"))
    )
    with pytest.raises(OSError):
        collection.checkpoint()
    crashed = tmp_path / "crashed"
    shutil.copytree(tmp_path / "collections", crashed / "collections")
    os.remove(crashed / "collections" / "c" / "LOCK")
    recovered = _open(crashed)
    assert recovered.recovery["replayed_records"] == 0
    assert recovered.get_vector_with_metadata("a") == ([2.0] * 8, {"n": 2})


def test_corrupt_graph_file_falls_back_to_rebuild(tmp_path):
    collection = _open(tmp_path)
    for i, vector in enumerate(_vectors(30)):
        collection.add_vector(f"v{i}", vector)
    collection.close()
    root = tmp_path / "collections" / "c"
    current = json.loads((root / "CURRENT").read_text())["current"]
    manifest = json.loads((root / current / "manifest.json").read_text())
    assert "hnsw.npz" in manifest["checksums"]
    # Valid checksum, structurally wrong graph: re-point the manifest at tampered content.
    import hashlib

    tampered = (root / current / "hnsw.json").read_bytes().replace(b'"nodes": 30', b'"nodes": 31')
    (root / current / "hnsw.json").write_bytes(tampered)
    manifest["checksums"]["hnsw.json"] = hashlib.sha256(tampered).hexdigest()
    (root / current / "manifest.json").write_text(json.dumps(manifest))
    reopened = _open(tmp_path)
    assert reopened.recovery["graph_restored"] is False
    assert len(reopened.hnsw_index.nodes) == 30
    reopened.verify()


def _flip_last_byte_of_current_vectors(tmp_path):
    root = tmp_path / "collections" / "c"
    current = json.loads((root / "CURRENT").read_text())["current"]
    path = root / current / "vectors.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0x01  # silent single-bit corruption
    path.write_bytes(bytes(data))


def test_bit_rot_in_newest_checkpoint_is_detected_and_refused_by_default(tmp_path):
    collection = _open(tmp_path)
    collection.add_vector("a", [1.0] * 8)
    collection.checkpoint()
    collection.add_vector("b", [2.0] * 8)
    collection.checkpoint()
    collection.close()
    _flip_last_byte_of_current_vectors(tmp_path)
    with pytest.raises(CorruptStoreError, match="allow_stale_checkpoint"):
        _open(tmp_path)
    reopened = _open(tmp_path, allow_stale_checkpoint=True)  # explicit, lossy
    assert list(reopened.store.vectors) == ["a"]
    assert reopened.recovery["stale_checkpoint"] is True
    reopened.add_vector("c", [3.0] * 8)  # and it keeps working afterwards
    reopened.close()
    assert set(_open(tmp_path).store.vectors) == {"a", "c"}


def test_log_that_does_not_connect_to_checkpoint_is_refused(tmp_path):
    collection = _open(tmp_path)
    collection.add_vector("a", [1.0] * 8)
    collection.checkpoint()
    collection.add_vector("b", [2.0] * 8)
    collection.checkpoint()
    collection.add_vector("c", [3.0] * 8)  # lives only in the log (lsn 3)
    crashed = tmp_path / "crashed"
    shutil.copytree(tmp_path / "collections", crashed / "collections")
    os.remove(crashed / "collections" / "c" / "LOCK")
    _flip_last_byte_of_current_vectors(crashed)
    with pytest.raises(CorruptStoreError):
        _open(crashed)
    lossy = _open(crashed, allow_stale_checkpoint=True)
    assert set(lossy.store.vectors) == {"a"}  # "c" depended on the lost "b"; not replayed


def test_only_two_generations_are_retained(tmp_path):
    collection = _open(tmp_path)
    for i in range(6):
        collection.add_vector(f"v{i}", [float(i + 1)] * 8)
        collection.checkpoint()
    generations = [
        p for p in (tmp_path / "collections" / "c").iterdir() if p.name.startswith("gen-")
    ]
    assert len(generations) == 2


def test_second_opener_is_refused(tmp_path):
    first = _open(tmp_path)
    with pytest.raises(StoreLockedError):
        _open(tmp_path)
    first.close()
    _open(tmp_path).close()


def test_closed_collection_rejects_operations(tmp_path):
    collection = _open(tmp_path)
    collection.close()
    with pytest.raises(ClosedError):
        collection.add_vector("a", [1.0] * 8)
    with pytest.raises(ClosedError):
        collection.search_similar([1.0] * 8)


def test_auto_checkpoint_bounds_log_growth(tmp_path):
    collection = _open(tmp_path, checkpoint_ops=10)
    for i in range(25):
        collection.add_vector(f"v{i}", [float(i + 1)] * 8)
    assert collection._ops_since_checkpoint < 10
    assert collection.health()["last_checkpoint_time"] is not None


def test_empty_collection_resets_dimension_and_index(tmp_path):
    collection = _open(tmp_path)
    collection.add_vector("a", [1.0, 2.0])
    collection.remove_vector("a")
    collection.add_vector("b", [1.0, 2.0, 3.0])
    assert collection.dimension == 3
    collection.verify()


def test_concurrent_mixed_workload_stays_consistent(tmp_path):
    collection = _open(tmp_path, checkpoint_ops=200)
    vectors = _vectors(400, seed=3)
    errors = []

    def writer(offset):
        try:
            for i in range(100):
                collection.add_vector(f"w{offset}-{i}", vectors[offset * 100 + i])
                if i % 5 == 4:
                    collection.remove_vector(f"w{offset}-{i - 2}")
        except Exception as exc:
            errors.append(exc)

    def reader():
        try:
            for i in range(150):
                for _id, score in collection.search_similar(vectors[i], k=5):
                    assert -1.0001 <= score <= 1.0001
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    collection.verify()
    expected = set(collection.store.vectors)
    collection.close()
    reopened = _open(tmp_path)
    assert set(reopened.store.vectors) == expected
    reopened.verify()


CRASH_CHILD = textwrap.dedent(
    """
    import os, signal, sys
    from vecstream import Collection
    c = Collection("c", sys.argv[1], True, {"M": 8, "ef_construction": 40}, checkpoint_ops=7)
    for i in range(int(sys.argv[2])):
        c.add_vector(f"v{i}", [float(i + 1), 1.0, 0.5, 2.0])
        print(i, flush=True)  # acknowledged only after add_vector returns
    os.kill(os.getpid(), signal.SIGKILL)
    """
)


@pytest.mark.parametrize("count", [1, 6, 7, 8, 20])
def test_sigkill_never_loses_acknowledged_writes(tmp_path, count):
    child = subprocess.run(
        [sys.executable, "-c", CRASH_CHILD, str(tmp_path), str(count)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )
    assert child.returncode == -signal.SIGKILL
    acknowledged = {f"v{int(line)}" for line in child.stdout.split()}
    assert len(acknowledged) == count
    recovered = Collection("c", str(tmp_path), True, {"M": 8, "ef_construction": 40})
    assert set(recovered.store.vectors) == acknowledged
    recovered.verify()
    assert recovered.search_similar([3.0, 1.0, 0.5, 2.0], k=1)[0][0] == "v2" or count < 3


def test_manager_rejects_path_traversal_and_cleans_up(tmp_path):
    manager = CollectionManager(str(tmp_path))
    for bad in ["../evil", "a/b", "", ".hidden", "x" * 200, "name."]:
        with pytest.raises(ValueError):
            manager.create_collection(bad)
    manager.create_collection("good").add_vector("a", [1.0, 0.0])
    manager.delete_collection("good")
    assert not (tmp_path / "collections" / "good").exists()
    manager.close()
    with pytest.raises(ClosedError):
        manager.create_collection("again")


def test_manager_reopen_restores_everything(tmp_path):
    manager = CollectionManager(str(tmp_path), sync=False)
    c = manager.create_collection("docs")
    c.add_vector("a", [1.0, 0.0], {"t": 1})
    manager.close()
    again = CollectionManager(str(tmp_path), sync=False)
    assert again.get_collection("docs").get_vector_with_metadata("a") == ([1.0, 0.0], {"t": 1})
    again.close()
