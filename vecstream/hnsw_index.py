"""A compact, from-scratch Hierarchical Navigable Small World index."""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

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
        if not ordered:
            return []
        matrix = np.stack([self._normalized_nodes[item_id] for item_id, _ in ordered])
        between = np.clip(1.0 - matrix @ matrix.T, 0.0, 2.0)
        selected: List[int] = []
        rejected: List[int] = []
        # Distance from each candidate to its nearest already-selected neighbor.
        nearest_selected = np.full(len(ordered), np.inf)
        for position, (_, query_distance) in enumerate(ordered):
            if nearest_selected[position] > query_distance:
                selected.append(position)
                np.minimum(nearest_selected, between[position], out=nearest_selected)
                if len(selected) == M:
                    return [ordered[p][0] for p in selected]
            else:
                rejected.append(position)
        selected.extend(rejected[: M - len(selected)])
        return [ordered[p][0] for p in selected]

    def _prune(self, node_id: str, level: int) -> None:
        graph = self.graphs[level]
        limit = self.M_max0 if level == 0 else self.M
        if len(graph[node_id]) <= limit:
            return
        neighbors = sorted(graph[node_id])
        matrix = np.stack([self._normalized_nodes[neighbor] for neighbor in neighbors])
        distances = np.clip(1.0 - matrix @ self._normalized_nodes[node_id], 0.0, 2.0)
        candidates = [
            (float(d), neighbor) for d, neighbor in zip(distances, neighbors, strict=True)
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
        """Remove a node, then repair the neighborhoods it leaves behind.

        Each former neighbor regains edges to the removed node's other
        neighbors (chosen by the usual diversity heuristic, never exceeding
        the degree limit), so deletions do not erode graph connectivity.
        """
        if id not in self.nodes:
            raise KeyError(f"Item with ID {id} not found in index")
        top = self.node_levels[id]
        orphans_by_level: Dict[int, Set[str]] = {}
        for level in range(top + 1):
            graph = self.graphs[level]
            neighbors = graph.pop(id, set())
            for neighbor in neighbors:
                graph[neighbor].discard(id)
            orphans_by_level[level] = neighbors
        del self.nodes[id]
        del self._normalized_nodes[id]
        del self.node_levels[id]
        if self.ep == id:
            self.ep = self._successor_entry(orphans_by_level[top])
        for level, orphans in orphans_by_level.items():
            self._repair(level, orphans)

    def _successor_entry(self, top_neighbors: Set[str]) -> Optional[str]:
        """Pick a new entry point at the maximum level, scanning only if needed."""
        if not self.nodes:
            return None
        if top_neighbors:  # neighbors on the removed entry's top layer share that level
            return min(top_neighbors)
        return max(sorted(self.node_levels), key=lambda item_id: self.node_levels[item_id])

    def _repair(self, level: int, orphans: Set[str]) -> None:
        """Reconnect nodes that lost an edge to a removed node.

        Each orphan gains at most as many edges as it has free degree, picked
        from the other orphans by the diversity heuristic against its existing
        neighbors. Existing edges are never dropped, so no node can be
        disconnected by repair.
        """
        graph = self.graphs[level]
        limit = self.M_max0 if level == 0 else self.M
        for node in sorted(orphans):
            existing = graph[node]
            room = limit - len(existing)
            pool = sorted(orphans - existing - {node})
            if room <= 0 or not pool:
                continue
            candidates = np.stack([self._normalized_nodes[other] for other in pool])
            to_node = np.clip(1.0 - candidates @ self._normalized_nodes[node], 0.0, 2.0)
            nearest = np.full(len(pool), np.inf)
            if existing:
                kept_matrix = np.stack([self._normalized_nodes[member] for member in existing])
                nearest = np.clip(1.0 - candidates @ kept_matrix.T, 0.0, 2.0).min(axis=1)
            gram = np.clip(1.0 - candidates @ candidates.T, 0.0, 2.0)
            chosen: List[int] = []
            skipped: List[int] = []
            for position in np.lexsort((np.arange(len(pool)), to_node)).tolist():
                if nearest[position] > to_node[position]:
                    chosen.append(position)
                    np.minimum(nearest, gram[position], out=nearest)
                else:
                    skipped.append(position)
                if len(chosen) == room:
                    break
            chosen.extend(skipped[: room - len(chosen)])
            added = [pool[position] for position in chosen]
            for other in added:
                graph[node].add(other)
                graph[other].add(node)
            for other in added:
                self._prune(other, level)

    def search(
        self,
        query: Sequence[float] | np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
        predicate: Optional[Callable[[str], bool]] = None,
        min_score: Optional[float] = None,
        max_visits: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Return the approximate top-``k`` as ``(id, cosine_similarity)``.

        ``predicate`` and ``min_score`` are applied *during* traversal: only
        qualifying nodes count toward the result set, while every node is
        still used to navigate. If the filter is so selective that the search
        visits ``max_visits`` nodes without finding ``k`` matches, the result
        is the exact top-``k`` among matching nodes (a full scan), so a
        filtered search never silently under-fills while matches exist.
        """
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
        ef = max(k, ef_search or k)
        if predicate is None and min_score is None:
            candidates = self._search_layer(q, entry, ef, 0)
            return [(item_id, float(1.0 - distance)) for distance, item_id in candidates[:k]]
        return self._search_filtered(q, entry, k, ef, predicate, min_score, max_visits)[0]

    def _search_filtered(
        self,
        q: np.ndarray,
        entry: str,
        k: int,
        ef: int,
        predicate: Optional[Callable[[str], bool]],
        min_score: Optional[float],
        max_visits: Optional[int],
    ) -> Tuple[List[Tuple[str, float]], int, bool]:
        """Return ``(results, nodes_visited, used_exact_fallback)``."""
        limit = max_visits if max_visits is not None else max(2000, 50 * ef)
        max_distance = float("inf") if min_score is None else 1.0 - min_score

        def qualifies(item_id: str, distance: float) -> bool:
            return distance <= max_distance and (predicate is None or predicate(item_id))

        graph = self.graphs[0]
        initial = self._normalized_distance(q, self._normalized_nodes[entry])
        candidates: List[Tuple[float, str]] = [(initial, entry)]
        results: List[Tuple[float, str]] = []
        if qualifies(entry, initial):
            results.append((-initial, entry))
        visited = {entry}
        exhausted = False
        while candidates:
            candidate_distance, candidate_id = heapq.heappop(candidates)
            if len(results) >= ef and candidate_distance > -results[0][0]:
                break
            for neighbor in graph.get(candidate_id, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                distance = self._normalized_distance(q, self._normalized_nodes[neighbor])
                if len(results) < ef or distance < -results[0][0]:
                    heapq.heappush(candidates, (distance, neighbor))
                    if qualifies(neighbor, distance):
                        heapq.heappush(results, (-distance, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)
            if len(visited) >= limit and len(results) < k:
                exhausted = True
                break
        if exhausted:
            return self._exact_filtered(q, k, qualifies), len(visited), True
        ordered = sorted((-negative, item_id) for negative, item_id in results)[:k]
        return (
            [(item_id, float(1.0 - distance)) for distance, item_id in ordered],
            len(visited),
            False,
        )

    def _exact_filtered(
        self, q: np.ndarray, k: int, qualifies: Callable[[str, float], bool]
    ) -> List[Tuple[str, float]]:
        scored = []
        for item_id, vector in self._normalized_nodes.items():
            distance = self._normalized_distance(q, vector)
            if qualifies(item_id, distance):
                scored.append((distance, item_id))
        scored.sort()
        return [(item_id, float(1.0 - distance)) for distance, item_id in scored[:k]]

    def stats(self) -> Dict[str, float]:
        level0 = self.graphs.get(0, {})
        degrees = [len(neighbors) for neighbors in level0.values()]
        return {
            "nodes": len(self.nodes),
            "levels": (self.node_levels[self.ep] + 1) if self.ep is not None else 0,
            "avg_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
            "isolated": sum(1 for d in degrees if d == 0) if len(degrees) > 1 else 0,
        }

    def export_topology(self, order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """Levels, directed ``(level, src, dst)`` edges, and entry position by ``order``."""
        position = {item_id: i for i, item_id in enumerate(order)}
        if set(position) != set(self.nodes):
            raise IndexInvariantError("index nodes differ from the vector store")
        levels = np.fromiter((self.node_levels[i] for i in order), dtype=np.int32, count=len(order))
        edges = [
            (level, position[src], position[dst])
            for level, graph in self.graphs.items()
            for src, neighbors in graph.items()
            for dst in neighbors
        ]
        entry = position[self.ep] if self.ep is not None else None
        return levels, np.asarray(edges, dtype=np.int32).reshape(-1, 3), entry

    @classmethod
    def from_topology(
        cls,
        dim: int,
        M: int,
        ef_construction: int,
        ml: int | None,
        ids: Sequence[str],
        vectors: Dict[str, np.ndarray],
        levels: np.ndarray,
        edges: np.ndarray,
        ep: int | None,
    ) -> "HNSWIndex":
        """Rebuild from ``export_topology`` output; raises ``IndexInvariantError`` if unsound."""
        index = cls(dim=dim, M=M, ef_construction=ef_construction, ml=ml)
        for i, item_id in enumerate(ids):
            raw = vectors[item_id]
            index.nodes[item_id] = raw
            index._normalized_nodes[item_id] = normalize(raw)
            index.node_levels[item_id] = int(levels[i])
        adjacency: Dict[int, Dict[str, Set[str]]] = {}
        for item_id, level in index.node_levels.items():
            for current in range(level + 1):
                adjacency.setdefault(current, {})[item_id] = set()
        try:
            for level, src, dst in edges.tolist():
                adjacency[level][ids[src]].add(ids[dst])
        except (KeyError, IndexError) as exc:
            raise IndexInvariantError(f"graph edge at an unpopulated node or level: {exc}") from exc
        index.graphs = adjacency
        index.ep = ids[ep] if ep is not None else None
        index.validate()
        return index

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
