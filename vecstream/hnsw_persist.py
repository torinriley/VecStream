"""Serialization of an HNSW graph so it can be restored without rebuilding."""

from __future__ import annotations

import io
import json
from typing import Dict, Mapping, Sequence, Set, Tuple

import numpy as np

from .errors import CorruptStoreError, IndexInvariantError
from .hnsw_index import HNSWIndex
from .vector_store import normalize

GRAPH_FORMAT = 1
META_FILE = "hnsw.json"
GRAPH_FILE = "hnsw.npz"


def dump_index(index: HNSWIndex, ids: Sequence[str]) -> Dict[str, bytes]:
    """Serialize graph topology; vectors stay in the vector checkpoint.

    ``ids`` is the vector store's ordering, so node numbers are positions in it.
    """
    position = {item_id: i for i, item_id in enumerate(ids)}
    if set(position) != set(index.nodes):
        raise IndexInvariantError("index nodes differ from the vector store")
    levels = np.fromiter((index.node_levels[i] for i in ids), dtype=np.int32, count=len(ids))
    edges = [
        (level, position[src], position[dst])
        for level, graph in index.graphs.items()
        for src, neighbors in graph.items()
        for dst in neighbors
    ]
    edge_array = np.asarray(edges, dtype=np.int32).reshape(-1, 3)
    buffer = io.BytesIO()
    np.savez(buffer, levels=levels, edges=edge_array)
    meta = {
        "graph_format": GRAPH_FORMAT,
        "dim": index.dim,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "ml": index.ml,
        "nodes": len(ids),
        "ep": position[index.ep] if index.ep is not None else None,
    }
    return {
        META_FILE: json.dumps(meta, sort_keys=True).encode("utf-8"),
        GRAPH_FILE: buffer.getvalue(),
    }


def load_index(
    files: Mapping[str, bytes], ids: Sequence[str], vectors: Mapping[str, np.ndarray]
) -> HNSWIndex:
    """Rebuild an ``HNSWIndex`` from persisted topology and the stored vectors.

    Raises ``CorruptStoreError`` if the graph is inconsistent with the vectors
    or violates any structural invariant; callers may then rebuild.
    """
    try:
        meta = json.loads(files[META_FILE])
        if meta["graph_format"] != GRAPH_FORMAT or meta["nodes"] != len(ids):
            raise CorruptStoreError("graph format or node count mismatch")
        with np.load(io.BytesIO(files[GRAPH_FILE]), allow_pickle=False) as archive:
            levels, edges = archive["levels"], archive["edges"]
        if levels.shape != (len(ids),) or edges.ndim != 2 or edges.shape[1] != 3:
            raise CorruptStoreError("graph arrays have the wrong shape")
        if len(ids) and (
            edges[:, 1:].min(initial=0) < 0 or edges[:, 1:].max(initial=0) >= len(ids)
        ):
            raise CorruptStoreError("graph edge references an unknown node")
        index = HNSWIndex(
            dim=meta["dim"], M=meta["M"], ef_construction=meta["ef_construction"], ml=meta["ml"]
        )
    except CorruptStoreError:
        raise
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise CorruptStoreError(f"persisted graph is unreadable: {exc}") from exc
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
    except KeyError as exc:
        raise CorruptStoreError(f"graph edge at an unpopulated level: {exc}") from exc
    index.graphs = adjacency
    index.ep = ids[meta["ep"]] if meta["ep"] is not None else None
    try:
        index.validate()
    except IndexInvariantError as exc:
        raise CorruptStoreError(f"persisted graph violates an invariant: {exc}") from exc
    return index


def graph_params(index: HNSWIndex) -> Tuple[int, int, int | None]:
    return index.M, index.ef_construction, index.ml
