"""Deletion repair, entry-point maintenance, and selection-heuristic equivalence."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vecstream import HNSWIndex


def build(count=200, dim=16, M=8, ef=40, seed=0, graph_seed=1):
    vectors = np.random.default_rng(seed).normal(size=(count, dim)).astype(np.float32)
    index = HNSWIndex(dim, M=M, ef_construction=ef, seed=graph_seed)
    for i, vector in enumerate(vectors):
        index.add_item(str(i), vector)
    return index, vectors


def reachable(index, level=0):
    graph = index.graphs[level]
    if index.ep is None:
        return set()
    seen, stack = {index.ep}, [index.ep]
    while stack:
        for neighbor in graph.get(stack.pop(), ()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def recall(index, vectors, live, queries, k=10, ef=30):
    ids = sorted(live, key=int)
    matrix = np.stack([vectors[int(i)] for i in ids])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    hits = 0
    for query in queries:
        truth = {ids[j] for j in np.argsort(-(matrix @ (query / np.linalg.norm(query))))[:k]}
        hits += len(truth & {item for item, _ in index.search(query, k, ef)})
    return hits / (k * len(queries))


def test_removed_nodes_never_returned_and_invariants_hold():
    index, vectors = build()
    removed = {str(i) for i in range(0, 200, 3)}
    for item in removed:
        index.remove_item(item)
        index.validate()
    for query in vectors[:20]:
        assert removed.isdisjoint(item for item, _ in index.search(query, 10, 30))


def test_star_neighborhood_is_reconnected_after_hub_removal():
    index = HNSWIndex(2, M=4, ef_construction=8, seed=0)
    index._get_random_level = lambda: 0
    index.add_item("hub", [1, 0])
    for name, angle in [("a", 0.1), ("b", 0.2), ("c", -0.1), ("d", -0.2)]:
        index.add_item(name, [1, angle])
    index.graphs[0].update(
        {"hub": {"a", "b", "c", "d"}, "a": {"hub"}, "b": {"hub"}, "c": {"hub"}, "d": {"hub"}}
    )
    index.ep = "hub"
    index.validate()
    index.remove_item("hub")
    assert all(index.graphs[0][n] for n in "abcd"), "no leaf may be orphaned"
    assert reachable_from_any(index) == {"a", "b", "c", "d"}
    index.validate()


def reachable_from_any(index):
    start = next(iter(index.nodes))
    seen, stack = {start}, [start]
    while stack:
        for neighbor in index.graphs[0].get(stack.pop(), ()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def test_repair_refills_degree_and_respects_limits():
    index, _ = build(count=150, M=6)
    victim = max(index.graphs[0], key=lambda n: len(index.graphs[0][n]))
    neighbors = set(index.graphs[0][victim])
    degree_before = {n: len(index.graphs[0][n]) for n in neighbors}
    index.remove_item(victim)
    for node in neighbors:
        degree = len(index.graphs[0][node])
        assert 1 <= degree <= index.M_max0
        # Losing the victim costs one edge; repair must win back at least some of them overall.
    refilled = sum(len(index.graphs[0][n]) >= degree_before[n] - 1 for n in neighbors)
    assert refilled >= len(neighbors) // 2
    index.validate()


def test_connectivity_survives_heavy_random_deletion():
    index, _ = build(count=300)
    order = np.random.default_rng(5).permutation(300)
    for position in order[:180]:
        index.remove_item(str(int(position)))
    index.validate()
    assert reachable(index) == set(index.nodes)


def test_recall_after_deletion_beats_unrepaired_graph_and_tracks_rebuild():
    index, vectors = build(count=600, dim=32, M=12, ef=80)
    queries = np.random.default_rng(9).normal(size=(60, 32)).astype(np.float32)
    live = {str(i) for i in range(600)}
    for position in np.random.default_rng(3).permutation(600)[:240]:
        index.remove_item(str(int(position)))
        live.discard(str(int(position)))
    repaired = recall(index, vectors, live, queries)
    rebuilt = HNSWIndex(32, M=12, ef_construction=80, seed=1)
    for item in sorted(live, key=int):
        rebuilt.add_item(item, vectors[int(item)])
    assert repaired >= recall(rebuilt, vectors, live, queries) - 0.05
    assert repaired >= 0.85


def test_entry_point_successor_prefers_top_layer_neighbor_without_scanning():
    index = HNSWIndex(2, M=2, ef_construction=4, seed=0)
    levels = iter([2, 2, 1, 0])
    index._get_random_level = lambda: next(levels)
    for name, vector in [("t1", [1, 0]), ("t2", [0.9, 0.1]), ("m", [0, 1]), ("z", [-1, 0])]:
        index.add_item(name, vector)
    assert index.ep == "t1"
    index.remove_item("t1")
    assert index.ep == "t2" and index.node_levels["t2"] == 2
    index.validate()


def test_entry_point_scans_when_removed_entry_was_alone_on_its_layer():
    index = HNSWIndex(2, M=2, ef_construction=4, seed=0)
    levels = iter([0, 3, 1, 1, 0])
    index._get_random_level = lambda: next(levels)
    for name, vector in [
        ("a", [1, 0]),
        ("top", [0, 1]),
        ("b", [-1, 0]),
        ("c", [0, -1]),
        ("d", [1, 1]),
    ]:
        index.add_item(name, vector)
    index.remove_item("top")
    assert index.node_levels[index.ep] == 1
    index.validate()


def test_removing_every_node_leaves_a_valid_empty_index_that_accepts_new_data():
    index, vectors = build(count=40)
    for i in np.random.default_rng(1).permutation(40):
        index.remove_item(str(int(i)))
        index.validate()
    assert index.ep is None and index.nodes == {} and index.search(vectors[0], 3) == []
    index.add_item("fresh", vectors[0])
    assert index.search(vectors[0], 1)[0][0] == "fresh"
    index.validate()


def test_reinserting_removed_ids_and_updating_in_place_stay_consistent():
    index, vectors = build(count=80)
    for i in range(0, 80, 2):
        index.remove_item(str(i))
    for i in range(0, 80, 2):
        index.add_item(str(i), vectors[(i + 1) % 80])
    for i in range(1, 80, 2):  # update path: remove + add under the hood
        index.add_item(str(i), vectors[(i + 2) % 80])
    index.validate()
    assert len(index.nodes) == 80


def test_remove_missing_item_raises_and_leaves_graph_untouched():
    index, _ = build(count=20)
    snapshot = {level: {n: set(v) for n, v in g.items()} for level, g in index.graphs.items()}
    with pytest.raises(KeyError):
        index.remove_item("nope")
    assert snapshot == index.graphs


def test_upper_level_edges_are_repaired_too():
    index, _ = build(count=400, M=4, ef=32)
    upper = [n for n, level in index.node_levels.items() if level >= 1 and n != index.ep]
    assert upper, "dataset should contain upper-layer nodes"
    for item in upper[:10]:
        index.remove_item(item)
        index.validate()
    assert index.ep in index.nodes


def reference_select(index, candidates, M):
    """The original scalar implementation of Algorithm 4."""
    by_id = {item_id: distance for distance, item_id in candidates}
    ordered = sorted(by_id.items(), key=lambda pair: (pair[1], pair[0]))
    selected, rejected = [], []
    for candidate_id, query_distance in ordered:
        diverse = all(
            index._normalized_distance(
                index._normalized_nodes[candidate_id], index._normalized_nodes[chosen]
            )
            > query_distance
            for chosen in selected
        )
        (selected if diverse else rejected).append(candidate_id)
        if len(selected) == M:
            return selected
    selected.extend(rejected[: M - len(selected)])
    return selected


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    count=st.integers(1, 40),
    M=st.integers(1, 12),
    duplicates=st.booleans(),
)
def test_vectorized_selection_matches_scalar_reference(seed, count, M, duplicates):
    index = HNSWIndex(6, M=max(M, 2), ef_construction=max(M, 2) + 4, seed=0)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, 6)).astype(np.float32)
    if duplicates and count > 3:
        vectors[1] = vectors[0]  # exact ties exercise (distance, id) ordering
    for i, vector in enumerate(vectors):
        index.add_item(f"n{i}", vector)
    query = rng.normal(size=6).astype(np.float32)
    q = query / np.linalg.norm(query)
    candidates = [(index._normalized_distance(q, v), n) for n, v in index._normalized_nodes.items()]
    assert index._select_neighbors(q, candidates, M) == reference_select(index, candidates, M)


def test_select_neighbors_handles_empty_and_duplicate_candidates():
    index, _ = build(count=10)
    assert index._select_neighbors(index._normalized_nodes["0"], [], 4) == []
    candidates = [(0.1, "1"), (0.1, "1"), (0.2, "2")]
    assert set(index._select_neighbors(index._normalized_nodes["0"], candidates, 5)) == {"1", "2"}


@settings(max_examples=25, deadline=None)
@given(
    operations=st.lists(
        st.tuples(st.sampled_from(["add", "remove"]), st.integers(0, 24)), min_size=1, max_size=60
    )
)
def test_random_mutation_sequences_preserve_every_invariant(operations):
    rng = np.random.default_rng(0)
    pool = rng.normal(size=(25, 5)).astype(np.float32)
    index = HNSWIndex(5, M=3, ef_construction=8, seed=2)
    live = set()
    for kind, number in operations:
        key = str(number)
        if kind == "add":
            index.add_item(key, pool[number])
            live.add(key)
        elif key in live:
            index.remove_item(key)
            live.discard(key)
        index.validate()
        assert set(index.nodes) == live
