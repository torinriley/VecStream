"""Backups: round trips, atomic publication, tamper detection, and restore safety."""

import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest

from vecstream import Collection, CollectionManager
from vecstream.backup import (
    INFO_FILE,
    default_backup_id,
    list_backups,
    validate_backup_id,
    verify_backup,
)
from vecstream.errors import CorruptStoreError, UnsupportedFormatVersionError

DIM = 12


def data(n=80, dim=DIM, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def populate(collection, n=80):
    collection.apply(
        [
            {"op": "put", "id": f"v{i}", "vector": v, "metadata": {"g": i % 4}}
            for i, v in enumerate(data(n))
        ]
    )


@pytest.fixture
def manager(tmp_path):
    manager = CollectionManager(str(tmp_path / "live"), sync=False, engine="python")
    yield manager
    manager.close()


def test_backup_restore_round_trip_preserves_search_metadata_and_graph(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source)
    query = data(1, seed=5)[0]
    expected = source.search_similar(query, 10, ef_search=60)
    info = source.backup(tmp_path / "bk")
    assert info["collection"] == "src" and info["vectors"] == 80 and info["dimension"] == DIM

    restored = manager.restore_collection("copy", tmp_path / "bk")
    assert restored.recovery["graph_restored"] is True  # the saved graph was reused
    assert restored.search_similar(query, 10, ef_search=60) == expected
    assert restored.get_vector_with_metadata("v3")[1] == {"g": 3}
    assert restored.verify()["vectors"] == 80
    restored.add_vector("new", [1.0] * DIM)  # the restored collection is fully writable
    assert restored.get_vector_count() == 81 and source.get_vector_count() == 80


def test_restored_collection_survives_a_restart(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source)
    source.backup(tmp_path / "bk")
    manager.restore_collection("copy", tmp_path / "bk").add_vector("late", [0.5] * DIM)
    manager.close()
    again = CollectionManager(str(tmp_path / "live"), sync=False, engine="python")
    try:
        assert again.get_collection("copy").get_vector_count() == 81
        assert again.collections_metadata["copy"]["restored_from"]["collection"] == "src"
    finally:
        again.close()


def test_backup_is_a_point_in_time_snapshot(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 40)
    source.backup(tmp_path / "bk")
    source.apply([{"op": "del", "id": f"v{i}"} for i in range(20)])
    restored = manager.restore_collection("copy", tmp_path / "bk")
    assert restored.get_vector_count() == 40 and source.get_vector_count() == 20


def test_unflushed_writes_are_included(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 10)
    source.add_vector("after-checkpoint", [1.0] * DIM)  # only in the write-ahead log so far
    source.backup(tmp_path / "bk")
    assert manager.restore_collection("copy", tmp_path / "bk").get_vector_count() == 11


def test_empty_collection_round_trips(manager, tmp_path):
    source = manager.create_collection("empty")
    source.backup(tmp_path / "bk")
    restored = manager.restore_collection("copy", tmp_path / "bk")
    assert restored.get_vector_count() == 0
    restored.add_vector("a", [1, 2, 3])
    assert restored.dimension == 3


def test_backup_does_not_stop_concurrent_traffic(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 120)
    errors, stop = [], threading.Event()

    def traffic():
        try:
            i = 0
            while not stop.is_set():
                source.search_similar(data(1, seed=i)[0], 3)
                source.add_vector(f"extra{i}", data(1, seed=i + 1000)[0])
                i += 1
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=traffic)
    thread.start()
    try:
        for n in range(4):
            source.backup(tmp_path / f"bk{n}")
    finally:
        stop.set()
        thread.join()
    assert errors == []
    for n in range(4):
        info = verify_backup(tmp_path / f"bk{n}")
        assert info["vectors"] >= 120
    source.verify()


def test_destination_must_not_exist_and_nothing_is_overwritten(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 10)
    source.backup(tmp_path / "bk")
    before = {p.name: p.read_bytes() for p in (tmp_path / "bk").iterdir()}
    with pytest.raises(FileExistsError):
        source.backup(tmp_path / "bk")
    assert {p.name: p.read_bytes() for p in (tmp_path / "bk").iterdir()} == before


def test_failed_backup_leaves_no_partial_directory_or_staging(manager, tmp_path, monkeypatch):
    source = manager.create_collection("src")
    populate(source, 10)

    def explode(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("vecstream.backup._copy_durably", explode)
    with pytest.raises(OSError):
        source.backup(tmp_path / "out" / "bk")
    assert list((tmp_path / "out").iterdir()) == []
    assert not list(Path(source.collection_dir).glob(".backup-*"))
    monkeypatch.undo()
    source.backup(tmp_path / "out" / "bk")  # and the collection still works afterwards


def test_interrupted_backup_debris_is_cleaned_on_open(tmp_path):
    first = Collection("c", str(tmp_path), sync=False, engine="python")
    populate(first, 5)
    first.close()
    debris = Path(first.collection_dir) / ".backup-deadbeef"
    debris.mkdir()
    (debris / "junk").write_text("x")
    Collection("c", str(tmp_path), sync=False, engine="python").close()
    assert not debris.exists()


@pytest.mark.parametrize("victim", ["vectors.npy", "ids.json", "metadata.json", "hnsw.npz"])
def test_any_flipped_byte_is_detected(manager, tmp_path, victim):
    source = manager.create_collection("src")
    populate(source, 30)
    source.backup(tmp_path / "bk")
    path = tmp_path / "bk" / victim
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    path.write_bytes(bytes(raw))
    with pytest.raises(CorruptStoreError):
        verify_backup(tmp_path / "bk")
    with pytest.raises(CorruptStoreError):
        manager.restore_collection("copy", tmp_path / "bk")
    assert "copy" not in manager.list_collections()
    assert not (Path(manager.collections_dir) / "copy").exists()


def test_missing_manifest_info_and_tampered_manifest_are_rejected(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 10)
    source.backup(tmp_path / "bk")
    shutil.copytree(tmp_path / "bk", tmp_path / "no-info")
    (tmp_path / "no-info" / INFO_FILE).unlink()
    with pytest.raises(CorruptStoreError):
        verify_backup(tmp_path / "no-info")
    shutil.copytree(tmp_path / "bk", tmp_path / "bad-manifest")
    manifest = tmp_path / "bad-manifest" / "manifest.json"
    tampered = json.loads(manifest.read_text())
    assert tampered["vector_count"] == 10
    tampered["vector_count"] = 11
    manifest.write_text(json.dumps(tampered))
    with pytest.raises(CorruptStoreError):
        verify_backup(tmp_path / "bad-manifest")
    shutil.copytree(tmp_path / "bk", tmp_path / "missing-file")
    (tmp_path / "missing-file" / "vectors.npy").unlink()
    with pytest.raises(CorruptStoreError):
        verify_backup(tmp_path / "missing-file")


def test_unsupported_backup_format_is_refused(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 5)
    source.backup(tmp_path / "bk")
    info_path = tmp_path / "bk" / INFO_FILE
    info = json.loads(info_path.read_text())
    info["backup_format"] = 99
    info_path.write_text(json.dumps(info))
    with pytest.raises(UnsupportedFormatVersionError):
        verify_backup(tmp_path / "bk")


def test_symlinked_member_is_refused(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 5)
    source.backup(tmp_path / "bk")
    outside = tmp_path / "outside.json"
    outside.write_text((tmp_path / "bk" / "ids.json").read_text())
    (tmp_path / "bk" / "ids.json").unlink()
    (tmp_path / "bk" / "ids.json").symlink_to(outside)
    with pytest.raises(CorruptStoreError):
        verify_backup(tmp_path / "bk")


def test_restore_refuses_existing_names_and_bad_names_without_side_effects(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 10)
    source.backup(tmp_path / "bk")
    with pytest.raises(ValueError, match="already exists"):
        manager.restore_collection("src", tmp_path / "bk")
    for bad in ("../escape", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            manager.restore_collection(bad, tmp_path / "bk")
    assert manager.list_collections() == ["src"]
    assert source.get_vector_count() == 10
    leftovers = [p.name for p in Path(manager.collections_dir).iterdir()]
    assert leftovers == ["src"]


def test_quantization_mismatch_fails_cleanly_and_can_be_overridden(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 10)
    source.backup(tmp_path / "bk")
    with pytest.raises(ValueError, match="native engine"):
        manager.restore_collection("q", tmp_path / "bk", hnsw_params={"quantization": "sq8"})
    assert "q" not in manager.list_collections()
    assert not (Path(manager.collections_dir) / "q").exists()
    again = manager.restore_collection("q", tmp_path / "bk", hnsw_params={"M": 8})
    assert again.hnsw_params == {"M": 8} and again.get_vector_count() == 10


def test_backup_survives_deleting_the_source_collection(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 25)
    source.backup(tmp_path / "bk")
    manager.delete_collection("src")
    assert manager.restore_collection("src", tmp_path / "bk").get_vector_count() == 25


def test_backup_ids_are_single_safe_segments():
    for ok in ("daily-2026-01-01", "a", "x.y_z"):
        assert validate_backup_id(ok) == ok
    for bad in ("", ".", "..", "../x", "a/b", "a\\b", ".hidden", "trailing.", "x" * 129, None, 7):
        with pytest.raises(ValueError):
            validate_backup_id(bad)  # type: ignore[arg-type]
    assert validate_backup_id(default_backup_id("docs"))


def test_list_backups_reports_only_complete_directories(manager, tmp_path):
    source = manager.create_collection("src")
    populate(source, 5)
    root = tmp_path / "root"
    source.backup(root / "one")
    (root / ".one.partial-x").mkdir()
    (root / "empty-dir").mkdir()
    rows = list_backups(root)
    assert [r["id"] for r in rows] == ["one"] and rows[0]["collection"] == "src"
    assert list_backups(tmp_path / "missing") == []
