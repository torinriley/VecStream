import math

import numpy as np

from vecstream.hnsw_index import HNSWIndex


def test_level_distribution_matches_geometric_tail():
    index = HNSWIndex(dim=2, M=8, seed=42)
    levels = np.array([index._get_random_level() for _ in range(50_000)])
    # Standard HNSW distribution has P(L >= 1) = 1/M.
    assert abs(float(np.mean(levels >= 1)) - 1 / 8) < 0.01
    assert index.level_multiplier == 1 / math.log(8)


def test_seed_reproduces_topology_and_degree_invariants():
    vectors = np.random.default_rng(7).normal(size=(250, 12)).astype(np.float32)
    first = HNSWIndex(12, M=8, ef_construction=40, seed=9)
    second = HNSWIndex(12, M=8, ef_construction=40, seed=9)
    for i, vector in enumerate(vectors):
        first.add_item(str(i), vector)
        second.add_item(str(i), vector)
    first.validate()
    second.validate()
    assert first.node_levels == second.node_levels
    assert first.graphs == second.graphs


def test_diversity_heuristic_does_not_only_choose_nearest():
    index = HNSWIndex(2, M=2, ef_construction=2, seed=1)
    # Two points lie on the same side of the query; the third is slightly
    # farther away but opens a different direction in the graph.
    for item_id, vector in {
        "near": [0.985, 0.174],
        "redundant": [0.978, 0.208],
        "diverse": [0.940, -0.342],
    }.items():
        index.nodes[item_id] = np.asarray(vector, dtype=np.float32)
        index._normalized_nodes[item_id] = index.nodes[item_id] / np.linalg.norm(
            index.nodes[item_id]
        )
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = [
        (index._normalized_distance(query, index._normalized_nodes[i]), i) for i in index.nodes
    ]
    assert index._select_neighbors(query, candidates, 2) == ["near", "diverse"]
