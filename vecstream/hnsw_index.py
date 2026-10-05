"""A compact, from-scratch Hierarchical Navigable Small World index."""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .errors import IndexInvariantError
from .vector_store import as_float32_vector, normalize


class HNSWIndex:
    """Approximate cosine search using the HNSW graph algorithm."""

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ml: int | None = None,
        seed: int | None = None,
    ) -> None:
        if dim < 1 or M < 2 or ef_construction < M:
            raise ValueError("dim >= 1, M >= 2, and ef_construction >= M are required")
        self.dim, self.M, self.M_max0 = dim, M, 2 * M
        self.ef_construction, self.ml = ef_construction, ml
        # P(level >= l) = exp(-l / multiplier) = M**(-l).
        self.level_multiplier = 1.0 / math.log(M)
        self._rng = np.random.default_rng(seed)
        self.nodes: Dict[str, np.ndarray] = {}
        self._normalized_nodes: Dict[str, np.ndarray] = {}
        self.node_levels: Dict[str, int] = {}
        self.graphs: Dict[int, Dict[str, Set[str]]] = (
            {level: {} for level in range(ml + 1)} if ml is not None else {}
        )
        self.ep: Optional[str] = None

    def _get_random_level(self) -> int:
        sample = max(float(self._rng.random()), np.finfo(float).tiny)
        level = int(-math.log(sample) * self.level_multiplier)
        return min(level, self.ml) if self.ml is not None else level

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        a_value = normalize(as_float32_vector(a, self.dim))
        b_value = normalize(as_float32_vector(b, self.dim))
        return self._normalized_distance(a_value, b_value)

    @staticmethod
    def _normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.clip(1.0 - np.dot(a, b), 0.0, 2.0))

    def _search_layer(self, q: np.ndarray, ep: str, ef: int, level: int) -> List[Tuple[float, str]]:
        if ef < 1 or ep not in self.nodes:
            return []
        initial = self._normalized_distance(q, self._normalized_nodes[ep])
        candidates: List[Tuple[float, str]] = [(initial, ep)]
        results: List[Tuple[float, str]] = [(-initial, ep)]
        visited = {ep}
        while candidates:
            candidate_distance, candidate_id = heapq.heappop(candidates)
            if len(results) >= ef and candidate_distance > -results[0][0]:
                break
            for neighbor in self.graphs.get(level, {}).get(candidate_id, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                distance = self._normalized_distance(q, self._normalized_nodes[neighbor])
                if len(results) < ef or distance < -results[0][0]:
                    heapq.heappush(candidates, (distance, neighbor))
                    heapq.heappush(results, (-distance, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
        return sorted([(-neg_distance, item_id) for neg_distance, item_id in results])

    def _select_neighbors(
        self, q: np.ndarray, candidates: List[Tuple[float, str]], M: int
    ) -> List[str]:
        """Select close but directionally diverse neighbors (Algorithm 4)."""
        by_id = {item_id: distance for distance, item_id in candidates}
        ordered = sorted(by_id.items(), key=lambda pair: (pair[1], pair[0]))
        selected: List[str] = []
        rejected: List[str] = []
        for candidate_id, query_distance in ordered:
            diverse = all(
                self._normalized_distance(
                    self._normalized_nodes[candidate_id], self._normalized_nodes[chosen]
                )
                > query_distance
                for chosen in selected
            )
            (selected if diverse else rejected).append(candidate_id)
            if len(selected) == M:
                return selected
        selected.extend(rejected[: M - len(selected)])
        return selected

    def _prune(self, node_id: str, level: int) -> None:
        graph = self.graphs[level]
        limit = self.M_max0 if level == 0 else self.M
        if len(graph[node_id]) <= limit:
            return
        candidates = [
            (
                self._normalized_distance(
                    self._normalized_nodes[node_id], self._normalized_nodes[neighbor]
                ),
                neighbor,
            )
            for neighbor in graph[node_id]
        ]
        keep = set(self._select_neighbors(self._normalized_nodes[node_id], candidates, limit))
        removed = graph[node_id] - keep
        graph[node_id] = keep
        for other in removed:
            graph.get(other, set()).discard(node_id)

    def add_item(self, id: str, vector: Sequence[float] | np.ndarray) -> None:
        item_id = str(id)
        raw_value = as_float32_vector(vector, self.dim).copy()
        value = normalize(raw_value)
        if item_id in self.nodes:
            self.remove_item(item_id)
        level = self._get_random_level()
        self.nodes[item_id], self._normalized_nodes[item_id] = raw_value, value
        self.node_levels[item_id] = level
        for current_level in range(level + 1):
            self.graphs.setdefault(current_level, {})[item_id] = set()
        if self.ep is None:
            self.ep = item_id
            return
        entry = self.ep
        entry_level = self.node_levels[entry]
        for current_level in range(entry_level, level, -1):
            result = self._search_layer(value, entry, 1, current_level)
            if result:
                entry = result[0][1]
        for current_level in range(min(level, entry_level), -1, -1):
            candidates = self._search_layer(value, entry, self.ef_construction, current_level)
            limit = self.M_max0 if current_level == 0 else self.M
            neighbors = self._select_neighbors(value, candidates, limit)
            graph = self.graphs[current_level]
            graph[item_id].update(neighbors)
            for neighbor in neighbors:
                graph[neighbor].add(item_id)
                self._prune(neighbor, current_level)
            if candidates:
                entry = candidates[0][1]
        if level > entry_level:
            self.ep = item_id

    def remove_item(self, id: str) -> None:
        if id not in self.nodes:
            raise KeyError(f"Item with ID {id} not found in index")
        for level in range(self.node_levels[id] + 1):
            graph = self.graphs[level]
            for neighbor in graph.get(id, set()):
                graph[neighbor].discard(id)
            graph.pop(id, None)
        del self.nodes[id]
        del self._normalized_nodes[id]
        del self.node_levels[id]
        self.ep = (
            max(self.node_levels, key=lambda item_id: self.node_levels[item_id])
            if self.nodes
            else None
        )

    def search(
        self, query: Sequence[float] | np.ndarray, k: int = 10, ef_search: int | None = None
    ) -> List[Tuple[str, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        q = normalize(as_float32_vector(query, self.dim))
        if self.ep is None:
            return []
        entry = self.ep
        for level in range(self.node_levels[entry], 0, -1):
            result = self._search_layer(q, entry, 1, level)
            if result:
                entry = result[0][1]
        candidates = self._search_layer(q, entry, max(k, ef_search or k), 0)
        return [(item_id, float(1.0 - distance)) for distance, item_id in candidates[:k]]

    def validate(self) -> None:
        """Raise ``IndexInvariantError`` with the first structural violation."""
        if bool(self.nodes) != (self.ep is not None):
            raise IndexInvariantError("entry point must exist iff graph is non-empty")
        if self.ep is not None and self.ep not in self.nodes:
            raise IndexInvariantError("entry point references a nonexistent node")
        if self.ep is not None:
            maximum_level = max(self.node_levels.values())
            if self.node_levels[self.ep] != maximum_level:
                raise IndexInvariantError(
                    f"entry point {self.ep!r} is at level {self.node_levels[self.ep]}, "
                    f"but maximum node level is {maximum_level}"
                )
        if set(self.nodes) != set(self.node_levels) or set(self.nodes) != set(
            self._normalized_nodes
        ):
            raise IndexInvariantError(
                "nodes, normalized nodes, and node_levels contain different IDs"
            )
        for item_id, max_level in self.node_levels.items():
            if max_level < 0:
                raise IndexInvariantError(f"node {item_id!r} has a negative level")
            for level in range(max_level + 1):
                if item_id not in self.graphs.get(level, {}):
                    raise IndexInvariantError(f"node {item_id!r} is missing from level {level}")
        for level, graph in self.graphs.items():
            limit = self.M_max0 if level == 0 else self.M
            for item_id, neighbors in graph.items():
                if item_id not in self.nodes or level > self.node_levels[item_id]:
                    raise IndexInvariantError(f"invalid node {item_id!r} at level {level}")
                if len(neighbors) > limit:
                    raise IndexInvariantError(
                        f"degree {len(neighbors)} exceeds {limit} at level {level}"
                    )
                for neighbor in neighbors:
                    if neighbor == item_id:
                        raise IndexInvariantError(
                            f"self-edge for node {item_id!r} at level {level}"
                        )
                    if neighbor not in self.nodes:
                        raise IndexInvariantError(f"edge references missing node {neighbor!r}")
                    if item_id not in graph.get(neighbor, set()):
                        raise IndexInvariantError(
                            f"edge {item_id!r}-{neighbor!r} is not bidirectional"
                        )
        for item_id, vector in self.nodes.items():
            if vector.shape != (self.dim,) or not np.all(np.isfinite(vector)):
                raise IndexInvariantError(
                    f"raw vector for node {item_id!r} must be finite with shape ({self.dim},)"
                )
            normalized = self._normalized_nodes[item_id]
            if normalized.shape != (self.dim,) or not np.all(np.isfinite(normalized)):
                raise IndexInvariantError(
                    f"normalized vector for node {item_id!r} is malformed or non-finite"
                )
            norm = float(np.linalg.norm(normalized.astype(np.float64)))
            if not (np.isclose(norm, 0.0, atol=1e-7) or np.isclose(norm, 1.0, atol=1e-5)):
                raise IndexInvariantError(
                    f"normalized vector for node {item_id!r} has norm {norm}, expected 0 or 1"
                )
