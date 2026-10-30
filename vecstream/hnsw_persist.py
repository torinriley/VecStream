"""Serialization of an HNSW graph so it can be restored without rebuilding.

The format is engine-agnostic (levels plus directed edges by node position), so
a checkpoint written by the Python engine loads in the native engine and back.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Mapping, Sequence, Type

import numpy as np

from .errors import CorruptStoreError, IndexInvariantError

GRAPH_FORMAT = 1
META_FILE = "hnsw.json"
GRAPH_FILE = "hnsw.npz"


def dump_index(index: Any, ids: Sequence[str]) -> Dict[str, bytes]:
    """Serialize graph topology; vectors stay in the vector checkpoint.

    ``ids`` is the vector store's ordering, so node numbers are positions in it.
    """
    levels, edges, entry = index.export_topology(ids)
    buffer = io.BytesIO()
    np.savez(buffer, levels=levels, edges=edges)
    meta = {
        "graph_format": GRAPH_FORMAT,
        "dim": index.dim,
        "M": index.M,
        "ef_construction": index.ef_construction,
        "ml": index.ml,
        "quantization": getattr(index, "quantization", "none"),
        "nodes": len(ids),
        "ep": entry,
    }
    return {
        META_FILE: json.dumps(meta, sort_keys=True).encode("utf-8"),
        GRAPH_FILE: buffer.getvalue(),
    }


def load_index(
    files: Mapping[str, bytes],
    ids: Sequence[str],
    vectors: Mapping[str, np.ndarray],
    index_class: Type[Any],
    quantization: str = "none",
) -> Any:
    """Rebuild an index of ``index_class`` from persisted topology and the stored vectors.

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
        index = index_class.from_topology(
            meta["dim"],
            meta["M"],
            meta["ef_construction"],
            meta["ml"],
            list(ids),
            dict(vectors),
            levels,
            edges,
            meta["ep"],
            quantization=quantization,
        )
    except CorruptStoreError:
        raise
    except IndexInvariantError as exc:
        raise CorruptStoreError(f"persisted graph violates an invariant: {exc}") from exc
    except (KeyError, ValueError, TypeError, OSError) as exc:
        raise CorruptStoreError(f"persisted graph is unreadable: {exc}") from exc
    return index
