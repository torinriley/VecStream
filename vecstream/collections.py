"""Durable, thread-safe collections built on a checkpoint plus write-ahead log.

Every mutation is validated, appended to the write-ahead log as one atomic
record (the commit point), and then applied to memory. Checkpoints write the
vectors, metadata, and HNSW graph as an immutable checksummed generation and
then reset the log. Recovery loads the newest valid generation, restores the
persisted graph, and replays committed log records on top of it.
"""

from __future__ import annotations

import base64
import binascii
import fcntl
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .binary_store import BinaryVectorStore
from .errors import ClosedError, CorruptStoreError, InvalidVectorError, StoreLockedError
from .hnsw_index import HNSWIndex
from .hnsw_persist import dump_index, load_index
from .rwlock import ReadWriteLock
from .vector_store import as_float32_vector
from .wal import WriteAheadLog, fsync_directory

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_ID_LENGTH = 512
MAX_METADATA_BYTES = 1 << 20
_HNSW_KEYS = {"M", "ef_construction", "ml"}


def validate_name(name: str) -> str:
    """Reject names that could escape the storage directory or confuse tooling."""
    if not isinstance(name, str) or not NAME_PATTERN.match(name) or name.endswith("."):
        raise ValueError(
            "name must be 1-128 characters of letters, digits, '_', '-', '.' "
            "and start with a letter or digit"
        )
    return name


def validate_hnsw_params(params: Dict[str, Any]) -> Dict[str, Any]:
    unknown = set(params) - _HNSW_KEYS
    if unknown:
        raise ValueError(f"unknown HNSW parameters: {sorted(unknown)}")
    m = params.get("M", 16)
    ef = params.get("ef_construction", 200)
    ml = params.get("ml")
    if not isinstance(m, int) or isinstance(m, bool) or m < 2:
        raise ValueError("M must be an integer >= 2")
    if not isinstance(ef, int) or isinstance(ef, bool) or ef < m:
        raise ValueError("ef_construction must be an integer >= M")
    if ml is not None and (not isinstance(ml, int) or isinstance(ml, bool) or ml < 0):
        raise ValueError("ml must be a non-negative integer or null")
    return dict(params)


def _encode_ops(ops: List[Dict[str, Any]]) -> bytes:
    encoded = []
    for op in ops:
        if op["op"] == "put":
            vector = np.asarray(op["vector"], dtype="<f4").tobytes()
            encoded.append(
                {
                    "op": "put",
                    "id": op["id"],
                    "v": base64.b64encode(vector).decode("ascii"),
                    "m": op["metadata"],
                }
            )
        else:
            encoded.append({"op": "del", "id": op["id"]})
    return json.dumps({"ops": encoded}, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode_ops(payload: bytes) -> List[Dict[str, Any]]:
    try:
        decoded = json.loads(payload)["ops"]
        ops: List[Dict[str, Any]] = []
        for item in decoded:
            if item["op"] == "put":
                vector = np.frombuffer(base64.b64decode(item["v"], validate=True), dtype="<f4")
                ops.append(
                    {
                        "op": "put",
                        "id": item["id"],
                        "vector": vector.astype(np.float32),
                        "metadata": item["m"],
                    }
                )
            elif item["op"] == "del":
                ops.append({"op": "del", "id": item["id"]})
            else:
                raise ValueError(f"unknown op {item['op']!r}")
        return ops
    except (KeyError, TypeError, ValueError, binascii.Error) as exc:
        raise CorruptStoreError(f"write-ahead log record cannot be decoded: {exc}") from exc


class Collection:
    """A named, durable, thread-safe set of vectors with optional HNSW search.

    Reads (``search_similar``, ``get_*``) run concurrently; mutations take an
    exclusive lock. Batches submitted through ``apply`` commit atomically.
    """

    def __init__(
        self,
        name: str,
        storage_dir: str,
        use_hnsw: bool = True,
        hnsw_params: Optional[Dict[str, Any]] = None,
        *,
        sync: bool = True,
        checkpoint_ops: int = 10_000,
        checkpoint_bytes: int = 64 << 20,
        wal_corruption: str = "raise",
        allow_stale_checkpoint: bool = False,
    ):
        """Open (recovering if necessary) or create a collection.

        Args:
            name: Collection name
            storage_dir: Base storage directory
            use_hnsw: Whether to use HNSW indexing (default: True)
            hnsw_params: Optional parameters for HNSW index
            sync: fsync the write-ahead log on every commit (disable only for tests)
            checkpoint_ops: Checkpoint automatically after this many logged operations
            checkpoint_bytes: ... or once the log reaches this size
            wal_corruption: ``"raise"`` or ``"truncate"`` on mid-log damage
            allow_stale_checkpoint: accept data loss by loading the previous
                checkpoint when the newest one is damaged (default: refuse)
        """
        self.name = name
        self.collection_dir = os.path.join(storage_dir, "collections", name)
        os.makedirs(self.collection_dir, exist_ok=True)
        self.use_hnsw = use_hnsw
        self.hnsw_params = validate_hnsw_params(hnsw_params or {})
        self.sync = sync
        self.checkpoint_ops = checkpoint_ops
        self.checkpoint_bytes = checkpoint_bytes
        self.wal_corruption = wal_corruption
        self.allow_stale_checkpoint = allow_stale_checkpoint
        self.hnsw_index: Optional[HNSWIndex] = None
        self._lock = ReadWriteLock()
        self._closed = False
        self._ops_since_checkpoint = 0
        self.last_checkpoint_time: Optional[float] = None
        self.checkpoint_failures = 0
        self.recovery: Dict[str, Any] = {}
        self._lock_fd = self._acquire_process_lock()
        try:
            self._open_state()
        except BaseException:
            self._release_process_lock()
            raise

    # -- lifecycle ---------------------------------------------------------------------------

    def _acquire_process_lock(self) -> int:
        descriptor = os.open(
            os.path.join(self.collection_dir, "LOCK"), os.O_RDWR | os.O_CREAT, 0o644
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(descriptor)
            raise StoreLockedError(
                f"collection {self.name!r} is already open in another process or object"
            ) from exc
        return descriptor

    def _release_process_lock(self) -> None:
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1

    def _open_state(self) -> None:
        """Load the checkpoint, restore the graph, and replay the log."""
        self.store = BinaryVectorStore(self.collection_dir, autosave=False)
        stale = (
            self.store.fell_back
            and self.store.pointer_lsn is not None
            and self.store.pointer_lsn > self.store.checkpoint_lsn
        )
        if stale and not self.allow_stale_checkpoint:
            raise CorruptStoreError(
                f"newest checkpoint of {self.name!r} is damaged; the previous one predates "
                f"operations up to lsn {self.store.pointer_lsn} (checkpoint is at lsn "
                f"{self.store.checkpoint_lsn}). Reopen with allow_stale_checkpoint=True to "
                "accept that data loss."
            )
        self.wal = WriteAheadLog(
            Path(self.collection_dir) / "wal.log",
            checkpoint_lsn=self.store.checkpoint_lsn,
            sync=self.sync,
            on_corruption=self.wal_corruption,
        )
        self.hnsw_index = None
        graph_restored = False
        if self.use_hnsw and self.store.dimension is not None:
            graph_restored = self._restore_graph()
            if self.hnsw_index is None:
                self._init_hnsw_index()
        records = self.wal.recovered_records
        if records and records[0][0] != self.store.checkpoint_lsn + 1:
            if not stale:
                raise CorruptStoreError(
                    f"write-ahead log starts at lsn {records[0][0]} but the checkpoint ends at "
                    f"{self.store.checkpoint_lsn}; operations are missing"
                )
            logger.error(
                "collection %s: discarding log that does not connect to checkpoint", self.name
            )
            records = []
            self.wal.reset(self.store.checkpoint_lsn)
        replayed = 0
        for _lsn, payload in records:
            self._apply_in_memory(_decode_ops(payload))
            replayed += 1
        self._ops_since_checkpoint = replayed
        self.recovery = {
            "checkpoint_lsn": self.store.checkpoint_lsn,
            "replayed_records": replayed,
            "torn_bytes_discarded": self.wal.truncated_bytes,
            "graph_restored": graph_restored,
            "stale_checkpoint": stale,
        }
        if replayed or self.wal.truncated_bytes:
            logger.warning("collection %s recovered: %s", self.name, self.recovery)

    def _restore_graph(self) -> bool:
        files = self.store.extra_files
        if "hnsw.json" not in files:
            return False
        try:
            index = load_index(files, list(self.store.vectors), self.store.vectors)
        except (CorruptStoreError, KeyError) as exc:
            logger.warning(
                "collection %s: persisted graph unusable, rebuilding: %s", self.name, exc
            )
            return False
        wanted = self._index_params()
        if (index.M, index.ef_construction, index.ml) != (
            wanted["M"],
            wanted["ef_construction"],
            wanted["ml"],
        ):
            return False
        self.hnsw_index = index
        return True

    def _index_params(self) -> Dict[str, Any]:
        return {
            "M": self.hnsw_params.get("M", 16),
            "ef_construction": self.hnsw_params.get("ef_construction", 200),
            "ml": self.hnsw_params.get("ml"),
        }

    def _init_hnsw_index(self) -> None:
        """Initialize the HNSW index with current vectors."""
        if self.store.dimension is None:
            return
        params = self._index_params()
        self.hnsw_index = HNSWIndex(dim=self.store.dimension, **params)
        for item_id, vector in self.store.vectors.items():
            self.hnsw_index.add_item(item_id, vector)

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError(f"collection {self.name!r} is closed")

    def close(self) -> None:
        """Checkpoint outstanding mutations and release the storage directory."""
        with self._lock.write():
            if self._closed:
                return
            try:
                if self._ops_since_checkpoint:
                    self._checkpoint_locked()
            finally:
                self._closed = True
                self.wal.close()
                self._release_process_lock()

    # -- mutation ----------------------------------------------------------------------------

    def _validate_ops(self, operations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check an entire batch against simulated state before anything is logged."""
        if not operations:
            raise ValueError("a batch must contain at least one operation")
        dimension = self.store.dimension
        live = set(self.store.vectors)
        checked: List[Dict[str, Any]] = []
        for operation in operations:
            kind = operation.get("op")
            item_id = operation.get("id")
            if not isinstance(item_id, str) or not item_id or len(item_id) > MAX_ID_LENGTH:
                raise ValueError(f"id must be a non-empty string of at most {MAX_ID_LENGTH} chars")
            if kind == "put":
                raw_vector = operation.get("vector")
                if raw_vector is None:
                    raise InvalidVectorError("a put operation requires a vector")
                vector = as_float32_vector(raw_vector, dimension)
                metadata = operation.get("metadata")
                if metadata is not None:
                    if not isinstance(metadata, dict):
                        raise ValueError("metadata must be an object or null")
                    try:
                        encoded = json.dumps(metadata, allow_nan=False)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"metadata is not valid JSON: {exc}") from exc
                    if len(encoded) > MAX_METADATA_BYTES:
                        raise ValueError("metadata exceeds the size limit")
                    metadata = json.loads(encoded)  # detach from caller-owned objects
                dimension = int(vector.size)
                live.add(item_id)
                checked.append(
                    {"op": "put", "id": item_id, "vector": vector.copy(), "metadata": metadata}
                )
            elif kind == "del":
                if item_id not in live:
                    raise KeyError(f"Vector with ID {item_id} not found")
                live.discard(item_id)
                if not live:
                    dimension = None
                checked.append({"op": "del", "id": item_id})
            else:
                raise ValueError(f"unknown operation {kind!r}; expected 'put' or 'del'")
        return checked

    def _apply_in_memory(self, ops: List[Dict[str, Any]]) -> None:
        for op in ops:
            if op["op"] == "put":
                self.store.put_in_memory(op["id"], op["vector"], op["metadata"])
                if self.use_hnsw:
                    if self.hnsw_index is None:
                        self._init_hnsw_index()
                    else:
                        self.hnsw_index.add_item(op["id"], op["vector"])
            else:
                self.store.delete_in_memory(op["id"])
                if self.hnsw_index is not None:
                    try:
                        self.hnsw_index.remove_item(op["id"])
                    except KeyError:
                        pass
                    if not self.store.vectors:
                        self.hnsw_index = None

    def apply(self, operations: Sequence[Dict[str, Any]]) -> int:
        """Atomically apply ``{"op": "put"|"del", ...}`` operations; returns the LSN.

        Either every operation becomes durable and visible, or none does. The
        log append is the commit point; a crash before it loses the batch, a
        crash after it replays the whole batch.
        """
        with self._lock.write():
            self._check_open()
            checked = self._validate_ops(operations)
            lsn = self.wal.append(_encode_ops(checked))
            try:
                self._apply_in_memory(checked)
            except BaseException:
                logger.exception("collection %s: apply failed after commit; reloading", self.name)
                self._reload_locked()
                raise
            self._ops_since_checkpoint += len(checked)
            if (
                self._ops_since_checkpoint >= self.checkpoint_ops
                or self.wal.size >= self.checkpoint_bytes
            ):
                try:
                    self._checkpoint_locked()
                except Exception:
                    self.checkpoint_failures += 1
                    logger.exception("collection %s: automatic checkpoint failed", self.name)
            return lsn

    def _reload_locked(self) -> None:
        self.wal.close()
        self._open_state()

    def add_vector(
        self, id: str, vector: Sequence[float], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Insert or replace a vector and its metadata."""
        self.apply([{"op": "put", "id": id, "vector": vector, "metadata": metadata}])

    def remove_vector(self, id: str) -> None:
        """Remove a vector; raises ``KeyError`` if it does not exist."""
        self.apply([{"op": "del", "id": id}])

    # -- checkpointing and integrity ---------------------------------------------------------

    def checkpoint(self) -> None:
        """Persist vectors, metadata, and the HNSW graph, then truncate the log."""
        with self._lock.write():
            self._check_open()
            self._checkpoint_locked()

    def _checkpoint_locked(self) -> None:
        extras: Dict[str, bytes] = {}
        if self.hnsw_index is not None:
            extras = dump_index(self.hnsw_index, list(self.store.vectors))
        lsn = self.wal.last_lsn
        self.store.save_checkpoint(extras, lsn=lsn)
        self.wal.reset(lsn)  # crash before this is harmless: replay skips lsn <= checkpoint
        self._ops_since_checkpoint = 0
        self.last_checkpoint_time = time.time()

    def verify(self) -> Dict[str, Any]:
        """Verify checkpoint checksums, graph invariants, and store/index agreement."""
        with self._lock.read():
            self._check_open()
            self.store.verify()
            if self.hnsw_index is not None:
                self.hnsw_index.validate()
                if set(self.hnsw_index.nodes) != set(self.store.vectors):
                    raise CorruptStoreError("index and vector store contain different IDs")
            return {"vectors": len(self.store.vectors), "wal_bytes": self.wal.size, "ok": True}

    # -- reads -------------------------------------------------------------------------------

    def search_similar(
        self,
        query: Sequence[float],
        k: int = 5,
        threshold: float = 0.0,
        ef_search: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors with optional filtering.

        Args:
            query: Query vector
            k: Number of results to return
            threshold: Minimum similarity score threshold
            ef_search: HNSW ef search parameter (only used with HNSW)
            filter_metadata: Optional metadata filtering criteria

        Returns:
            List of (id, similarity) tuples
        """
        with self._lock.read():
            self._check_open()
            return self._search_locked(query, k, threshold, ef_search, filter_metadata)

    def search_batch(
        self,
        queries: Sequence[Sequence[float]],
        k: int = 5,
        threshold: float = 0.0,
        ef_search: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Run many queries against one consistent snapshot.

        All queries are validated before any is executed, and all see the same
        state (the read lock is held throughout). Without HNSW or filters the
        exact path is a single matrix product. HNSW traversal is Python-bound
        and threads do not speed it up, so those queries run sequentially.
        """
        with self._lock.read():
            self._check_open()
            if len(queries) == 0:
                raise ValueError("a batch must contain at least one query")
            if k < 1:
                raise ValueError("k must be at least 1")
            for query in queries:
                as_float32_vector(query, self.store.dimension)
            if not (self.use_hnsw and self.hnsw_index is not None) and not filter_metadata:
                return self.store.search_batch(queries, k=k, threshold=threshold)
            return [
                self._search_locked(query, k, threshold, ef_search, filter_metadata)
                for query in queries
            ]

    def search_detailed(
        self,
        query: Sequence[float],
        k: int = 5,
        threshold: float = 0.0,
        ef_search: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Like ``search_similar`` but returns metadata read under the same lock."""
        with self._lock.read():
            self._check_open()
            results = self._search_locked(query, k, threshold, ef_search, filter_metadata)
            return [
                (item_id, score, self.store.metadata.get(item_id)) for item_id, score in results
            ]

    def _search_locked(
        self,
        query: Sequence[float],
        k: int,
        threshold: float,
        ef_search: Optional[int],
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        # If using HNSW index
        if self.use_hnsw and self.hnsw_index is not None:
            if filter_metadata or threshold > 0.0:
                # Constraints are enforced during graph traversal, not after it.
                predicate = self._metadata_predicate(filter_metadata)
                return self.hnsw_index.search(
                    query, k=k, ef_search=ef_search, predicate=predicate, min_score=threshold
                )
            results = self.hnsw_index.search(query, k=k, ef_search=ef_search)
            return [(id, sim) for id, sim in results if sim >= threshold][:k]
        else:
            # Use regular search from store
            if filter_metadata:
                # Need to manually filter results
                all_results = self.store.search_similar(
                    query, k=len(self.store.vectors), threshold=threshold
                )
                filtered_results = []

                for id, similarity in all_results:
                    _, metadata = self.store.get_vector_with_metadata(id)
                    if metadata and self._matches_filter(metadata, filter_metadata):
                        filtered_results.append((id, similarity))
                        if len(filtered_results) >= k:
                            break

                return filtered_results[:k]
            else:
                # No filtering needed
                return self.store.search_similar(query, k=k, threshold=threshold)

    def _metadata_predicate(
        self, filter_metadata: Optional[Dict[str, Any]]
    ) -> Optional[Callable[[str], bool]]:
        """Predicate over IDs; vectors without metadata never match a non-empty filter."""
        if not filter_metadata:
            return None
        metadata = self.store.metadata

        def predicate(item_id: str) -> bool:
            record = metadata.get(item_id)
            return record is not None and self._matches_filter(record, filter_metadata)

        return predicate

    def _matches_filter(self, metadata: Dict[str, Any], filter_query: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter query.

        Args:
            metadata: Vector metadata
            filter_query: Filter criteria

        Returns:
            True if metadata matches filter, False otherwise
        """
        for key, value in filter_query.items():
            # Handle nested keys with dot notation (e.g., "user.name")
            if "." in key:
                parts = key.split(".")
                current = metadata
                match = True

                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        match = False
                        break
                    current = current[part]

                if not match or parts[-1] not in current:
                    return False

                # Check value comparison for the last part
                if current[parts[-1]] != value:
                    return False
            else:
                # Simple key-value match
                if key not in metadata or metadata[key] != value:
                    return False

        return True

    def get_vector_with_metadata(self, id: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
        """Get a vector and its metadata.

        Args:
            id: Vector identifier

        Returns:
            Tuple of (vector, metadata)
        """
        with self._lock.read():
            self._check_open()
            return self.store.get_vector_with_metadata(id)

    def get_vector_count(self) -> int:
        """Get the number of vectors in the collection."""
        with self._lock.read():
            return len(self.store.vectors)

    @property
    def dimension(self) -> Optional[int]:
        return self.store.dimension

    def health(self) -> Dict[str, Any]:
        """Cheap operational snapshot of the index and durability state."""
        with self._lock.read():
            index = self.hnsw_index
            level0 = index.graphs.get(0, {}) if index else {}
            degrees = [len(neighbors) for neighbors in level0.values()]
            return {
                "vectors": len(self.store.vectors),
                "dimension": self.store.dimension,
                "wal_bytes": self.wal.size,
                "ops_since_checkpoint": self._ops_since_checkpoint,
                "last_lsn": self.wal.last_lsn,
                "last_checkpoint_time": self.last_checkpoint_time,
                "checkpoint_failures": self.checkpoint_failures,
                "index_nodes": len(index.nodes) if index else 0,
                "index_levels": (index.node_levels[index.ep] + 1) if index and index.ep else 0,
                "index_avg_degree": (sum(degrees) / len(degrees)) if degrees else 0.0,
                "index_isolated_nodes": sum(1 for d in degrees if d == 0)
                if len(degrees) > 1
                else 0,
                "vector_bytes": sum(v.nbytes for v in self.store.vectors.values()),
                "recovery": dict(self.recovery),
            }


class CollectionManager:
    """Thread-safe registry of collections under one storage directory."""

    def __init__(
        self,
        base_storage_dir: str,
        use_hnsw: bool = True,
        default_hnsw_params: Optional[Dict[str, Any]] = None,
        **collection_options: Any,
    ):
        """Initialize the collection manager.

        Args:
            base_storage_dir: Base directory for storing collections
            use_hnsw: Whether to use HNSW indexing by default
            default_hnsw_params: Default parameters for HNSW index
            collection_options: Durability options forwarded to ``Collection``
        """
        self.base_storage_dir = base_storage_dir
        self.collections_dir = os.path.join(base_storage_dir, "collections")
        self.metadata_file = os.path.join(base_storage_dir, "collections_metadata.json")
        os.makedirs(self.collections_dir, exist_ok=True)
        self.use_hnsw = use_hnsw
        self.default_hnsw_params = validate_hnsw_params(
            default_hnsw_params or {"M": 16, "ef_construction": 200}
        )
        self.collection_options = collection_options
        self._lock = threading.RLock()
        self._closed = False
        self.collections: Dict[str, Collection] = {}
        self.collections_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load collections metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if not isinstance(loaded, dict):
                    raise ValueError("top-level value must be an object")
                self.collections_metadata = loaded
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise CorruptStoreError(
                    f"collection metadata {self.metadata_file!r} is invalid: {exc}"
                ) from exc

    def _save_metadata(self) -> None:
        """Atomically and durably save collections metadata."""
        descriptor, temporary = tempfile.mkstemp(
            dir=self.base_storage_dir, prefix=".collections-metadata-"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(self.collections_metadata, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.metadata_file)
            fsync_directory(Path(self.base_storage_dir))
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise ClosedError("collection manager is closed")

    def create_collection(
        self,
        name: str,
        use_hnsw: Optional[bool] = None,
        hnsw_params: Optional[Dict[str, Any]] = None,
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Collection name
            use_hnsw: Whether to use HNSW indexing (defaults to manager setting)
            hnsw_params: Custom HNSW parameters (defaults to manager settings)

        Returns:
            The created collection

        Raises:
            ValueError: If the name is invalid or the collection already exists
        """
        validate_name(name)
        with self._lock:
            self._check_open()
            if name in self.collections or name in self.collections_metadata:
                raise ValueError(f"Collection '{name}' already exists")
            use_hnsw_flag = self.use_hnsw if use_hnsw is None else use_hnsw
            hnsw_params_dict = (
                dict(self.default_hnsw_params) if hnsw_params is None else dict(hnsw_params)
            )
            validate_hnsw_params(hnsw_params_dict)
            # A directory without metadata is debris from a crashed create or delete.
            stale = os.path.join(self.collections_dir, name)
            if os.path.exists(stale):
                shutil.rmtree(stale)
            collection = Collection(
                name,
                self.base_storage_dir,
                use_hnsw_flag,
                hnsw_params_dict,
                **self.collection_options,
            )
            self.collections_metadata[name] = {
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "use_hnsw": use_hnsw_flag,
                "hnsw_params": hnsw_params_dict,
            }
            try:
                self._save_metadata()
            except BaseException:
                del self.collections_metadata[name]
                collection.close()
                shutil.rmtree(stale, ignore_errors=True)
                raise
            self.collections[name] = collection
            return collection

    def get_collection(self, name: str) -> Collection:
        """Get a collection by name, opening (and recovering) it if needed.

        Raises:
            KeyError: If collection doesn't exist
        """
        with self._lock:
            self._check_open()
            if name in self.collections:
                return self.collections[name]
            if name not in self.collections_metadata:
                raise KeyError(f"Collection '{name}' does not exist")
            metadata = self.collections_metadata[name]
            collection = Collection(
                name,
                self.base_storage_dir,
                metadata.get("use_hnsw", self.use_hnsw),
                metadata.get("hnsw_params", self.default_hnsw_params),
                **self.collection_options,
            )
            self.collections[name] = collection
            return collection

    def list_collections(self) -> List[str]:
        """List all available collections."""
        with self._lock:
            return list(self.collections_metadata.keys())

    def delete_collection(self, name: str) -> None:
        """Delete a collection.

        Raises:
            KeyError: If collection doesn't exist
        """
        with self._lock:
            self._check_open()
            if name not in self.collections_metadata:
                raise KeyError(f"Collection '{name}' does not exist")
            collection = self.collections.pop(name, None)
            if collection is not None:
                collection.close()  # waits for in-flight operations to drain
            del self.collections_metadata[name]
            self._save_metadata()  # commit point; leftover files are debris
            collection_dir = os.path.join(self.collections_dir, name)
            if os.path.exists(collection_dir):
                shutil.rmtree(collection_dir)

    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a collection.

        Raises:
            KeyError: If collection doesn't exist
        """
        collection = self.get_collection(name)
        vectors_size, metadata_size = collection.store.get_store_size()
        return {
            "name": name,
            "vector_count": collection.get_vector_count(),
            "vectors_size_bytes": vectors_size,
            "metadata_size_bytes": metadata_size,
            "total_size_bytes": vectors_size + metadata_size,
            "using_hnsw": collection.use_hnsw,
            "dimension": collection.dimension,
            **self.collections_metadata.get(name, {}),
        }

    def open_collections(self) -> Iterator[Collection]:
        with self._lock:
            return iter(list(self.collections.values()))

    def checkpoint_all(self) -> None:
        for collection in self.open_collections():
            collection.checkpoint()

    def close(self) -> None:
        """Checkpoint and release every open collection."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            for collection in list(self.collections.values()):
                try:
                    collection.close()
                except Exception as exc:  # keep closing the rest, then report
                    errors.append(exc)
            self.collections.clear()
            if errors:
                raise errors[0]
