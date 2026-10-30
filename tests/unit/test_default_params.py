"""New collections default to M=32; anything already saved keeps its meaning."""

import json

import numpy as np
import pytest

from vecstream import Collection, CollectionManager
from vecstream.collections import DEFAULT_HNSW_PARAMS, LEGACY_HNSW_PARAMS


def vectors(n=40, dim=8, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def fill(collection, n=40):
    collection.apply([{"op": "put", "id": f"v{i}", "vector": v} for i, v in enumerate(vectors(n))])


def test_documented_defaults():
    assert DEFAULT_HNSW_PARAMS == {"M": 32, "ef_construction": 200}
    assert LEGACY_HNSW_PARAMS == {"M": 16, "ef_construction": 200}


def test_manager_creates_collections_with_m32_and_saves_complete_params(tmp_path):
    manager = CollectionManager(str(tmp_path), sync=False)
    collection = manager.create_collection("a")
    assert collection.hnsw_params == {"M": 32, "ef_construction": 200}
    assert manager.collections_metadata["a"]["hnsw_params"] == {"M": 32, "ef_construction": 200}
    manager.close()
    saved = json.loads((tmp_path / "collections_metadata.json").read_text())
    assert saved["a"]["hnsw_params"] == {"M": 32, "ef_construction": 200}


def test_partial_params_at_creation_are_completed_from_manager_defaults(tmp_path):
    manager = CollectionManager(str(tmp_path), sync=False)
    only_ef = manager.create_collection("a", hnsw_params={"ef_construction": 300})
    only_m = manager.create_collection("b", hnsw_params={"M": 8})
    assert only_ef.hnsw_params == {"M": 32, "ef_construction": 300}
    assert only_m.hnsw_params == {"M": 8, "ef_construction": 200}
    custom = CollectionManager(
        str(tmp_path / "other"), default_hnsw_params={"M": 20, "ef_construction": 100}, sync=False
    )
    assert custom.create_collection("c").hnsw_params == {"M": 20, "ef_construction": 100}
    with pytest.raises(ValueError):
        manager.create_collection("bad", hnsw_params={"M": 300})  # ef_construction 200 < M


def test_the_default_dict_is_never_shared_or_mutated(tmp_path):
    first = Collection("a", str(tmp_path), sync=False)
    first.hnsw_params["M"] = 99
    assert DEFAULT_HNSW_PARAMS["M"] == 32
    assert Collection("b", str(tmp_path), sync=False).hnsw_params["M"] == 32
    manager = CollectionManager(str(tmp_path / "m"), sync=False)
    manager.create_collection("x").hnsw_params["M"] = 77
    assert manager.default_hnsw_params["M"] == 32


def test_collections_with_saved_m16_params_keep_m16_and_their_graph(tmp_path):
    first = Collection("c", str(tmp_path), True, {"M": 16, "ef_construction": 200}, sync=False)
    fill(first)
    first.close()
    reopened = Collection("c", str(tmp_path), True, {"M": 16, "ef_construction": 200}, sync=False)
    assert reopened.recovery["graph_restored"] is True
    assert reopened.hnsw_index.M == 16


def test_partial_saved_params_keep_their_historical_meaning(tmp_path):
    """A store written before M was always saved: {'ef_construction': 300} meant M=16."""
    first = Collection("c", str(tmp_path), True, {"ef_construction": 300}, sync=False)
    assert first.hnsw_index is None and first._index_params() == {
        "M": 16,
        "ef_construction": 300,
        "ml": None,
    }
    fill(first)
    first.close()
    reopened = Collection("c", str(tmp_path), True, {"ef_construction": 300}, sync=False)
    assert reopened.recovery["graph_restored"] is True
    assert reopened.hnsw_index.M == 16


def test_stores_without_any_saved_params_load_as_m16_through_the_manager(tmp_path):
    legacy = Collection("old", str(tmp_path), True, {"M": 16, "ef_construction": 200}, sync=False)
    fill(legacy)
    legacy.close()
    (tmp_path / "collections_metadata.json").write_text(
        json.dumps({"old": {"name": "old", "use_hnsw": True}})  # no hnsw_params key at all
    )
    manager = CollectionManager(str(tmp_path), sync=False)
    old = manager.get_collection("old")
    assert old.recovery["graph_restored"] is True and old.hnsw_index.M == 16
    manager.close()


def test_changing_saved_params_rebuilds_the_graph_rather_than_trusting_it(tmp_path):
    first = Collection("c", str(tmp_path), True, {"M": 16, "ef_construction": 200}, sync=False)
    fill(first)
    first.close()
    changed = Collection("c", str(tmp_path), True, {"M": 24, "ef_construction": 200}, sync=False)
    assert changed.recovery["graph_restored"] is False and changed.hnsw_index.M == 24
    changed.verify()


def test_new_default_collection_round_trips_and_searches(tmp_path):
    manager = CollectionManager(str(tmp_path), sync=False)
    collection = manager.create_collection("a")
    fill(collection, 60)
    manager.close()
    again = CollectionManager(str(tmp_path), sync=False).get_collection("a")
    assert again.recovery["graph_restored"] is True and again.hnsw_index.M == 32
    assert again.search_similar(vectors(60)[3], k=1)[0][0] == "v3"
