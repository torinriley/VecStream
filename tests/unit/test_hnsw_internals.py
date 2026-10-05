import numpy as np
import pytest

from vecstream.errors import IndexInvariantError
from vecstream.hnsw_index import HNSWIndex
from vecstream.vector_store import normalize


def make_manual_index() -> HNSWIndex:
    index = HNSWIndex(2, M=2, ef_construction=4, seed=1)
    vectors = {"a": [1, 0], "b": [0.9, 0.1], "c": [0, 1], "d": [-1, 0]}
    for item_id, vector in vectors.items():
        raw = np.asarray(vector, dtype=np.float32)
        index.nodes[item_id] = raw
        index._normalized_nodes[item_id] = normalize(raw)
        index.node_levels[item_id] = 0
    index.graphs = {0: {"a": {"b"}, "b": {"a", "c"}, "c": {"b", "d"}, "d": {"c"}}}
    index.ep = "a"
    return index


def test_search_layer_orders_and_bounds_results():
    index = make_manual_index()
    query = normalize(np.asarray([-1, 0], dtype=np.float32))
    one = index._search_layer(query, "a", ef=1, level=0)
    many = index._search_layer(query, "a", ef=3, level=0)
    assert one == [(0.0, "d")]
    assert len(many) == 3
    assert many == sorted(many)
    assert many[0] == (0.0, "d")


def test_search_layer_handles_sparse_or_missing_level():
    index = make_manual_index()
    query = normalize(np.asarray([1, 0], dtype=np.float32))
    assert index._search_layer(query, "a", ef=4, level=7) == [(0.0, "a")]
    assert index._search_layer(query, "missing", ef=4, level=0) == []


@pytest.mark.parametrize("candidate_count", [1, 2, 5])
def test_neighbor_selection_respects_capacity(candidate_count):
    index = HNSWIndex(2, M=2, ef_construction=4, seed=1)
    for i in range(candidate_count):
        angle = i * 0.4
        raw = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
        index.nodes[str(i)] = raw
        index._normalized_nodes[str(i)] = normalize(raw)
    query = np.asarray([1, 0], dtype=np.float32)
    candidates = [
        (index._normalized_distance(query, index._normalized_nodes[str(i)]), str(i))
        for i in range(candidate_count)
    ]
    selected = index._select_neighbors(query, candidates, M=2)
    assert len(selected) == min(candidate_count, 2)
    assert len(selected) == len(set(selected))


def test_neighbor_selection_handles_duplicate_and_equidistant_vectors():
    index = HNSWIndex(2, M=2, ef_construction=4, seed=1)
    for item_id, raw in {"a": [1, 1], "b": [1, 1], "c": [1, -1]}.items():
        index.nodes[item_id] = np.asarray(raw, dtype=np.float32)
        index._normalized_nodes[item_id] = normalize(index.nodes[item_id])
    query = np.asarray([1, 0], dtype=np.float32)
    candidates = [
        (index._normalized_distance(query, index._normalized_nodes[i]), i) for i in index.nodes
    ]
    assert index._select_neighbors(query, candidates, 2) == ["a", "c"]


def test_pruning_removes_reciprocal_edges():
    index = HNSWIndex(2, M=2, ef_construction=4, seed=1)
    center = np.asarray([1, 0], dtype=np.float32)
    for item_id, raw in {"x": center, "a": [1, 0.1], "b": [0, 1], "c": [-1, 0]}.items():
        index.nodes[item_id] = np.asarray(raw, dtype=np.float32)
        index._normalized_nodes[item_id] = normalize(index.nodes[item_id])
        index.node_levels[item_id] = 1
    index.graphs = {
        0: {item_id: set() for item_id in index.nodes},
        1: {item_id: set() for item_id in index.nodes},
    }
    index.ep = "x"
    index.graphs[1]["x"] = {"a", "b", "c"}
    for neighbor in ("a", "b", "c"):
        index.graphs[1][neighbor].add("x")
    index._prune("x", 1)
    removed = {"a", "b", "c"} - index.graphs[1]["x"]
    assert len(index.graphs[1]["x"]) <= index.M
    assert all("x" not in index.graphs[1][node] for node in removed)
    index.validate()


def test_entry_point_lifecycle_and_update():
    index = HNSWIndex(2, M=2, ef_construction=4, seed=4)
    index._get_random_level = iter([0, 2, 0, 1]).__next__
    index.add_item("first", [1, 0])
    assert index.ep == "first"
    index.add_item("high", [0, 1])
    assert index.ep == "high"
    index.add_item("ordinary", [-1, 0])
    assert index.ep == "high"
    index.remove_item("ordinary")
    assert index.ep == "high"
    index.remove_item("high")
    assert index.ep == "first"
    index.add_item("first", [0, -1])
    assert index.nodes["first"].tolist() == [0.0, -1.0]
    index.validate()
    index.remove_item("first")
    assert index.ep is None
    index.validate()


def test_validate_rejects_self_edge_and_nonmax_entry_point():
    index = make_manual_index()
    index.graphs[0]["a"].add("a")
    with pytest.raises(IndexInvariantError, match="self-edge"):
        index.validate()
    index.graphs[0]["a"].remove("a")
    index.node_levels["d"] = 1
    index.graphs[1] = {"d": set()}
    with pytest.raises(IndexInvariantError, match="entry point.*maximum"):
        index.validate()
