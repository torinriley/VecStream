"""Native HNSW engine bridge (Rust, loaded with ctypes).

The library is optional: ``native_available()`` is False when it has not been
built, and everything falls back to the pure-Python ``HNSWIndex``. Build it with
``python -m vecstream.native build`` (requires a Rust toolchain).

ctypes releases the GIL for the duration of each foreign call, so concurrent
searches on one index run in parallel. The wrapper and the native handle each
carry a readers-writer lock: searches share, mutations are exclusive, and misuse
from several threads blocks rather than races. A predicate callback must not call
back into a mutating method of the same index (it would deadlock).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import CorruptStoreError, IndexInvariantError
from .rwlock import ReadWriteLock
from .vector_store import as_float32_vector, normalize

ABI_VERSION = 2
MAX_DIM = 65_536  # mirrors the limits enforced inside the native library
MAX_M = 1_024
MAX_EF = 1_000_000
_LIB_NAME = {"darwin": "libvecstream_native.dylib", "win32": "vecstream_native.dll"}.get(
    sys.platform, "libvecstream_native.so"
)
_ROOT = Path(__file__).resolve().parent.parent
_PRED = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_uint32)
_lock = threading.Lock()
_library: Optional[ctypes.CDLL] = None
_load_error: Optional[str] = None


def _candidates() -> List[Path]:
    paths = []
    if os.environ.get("VECSTREAM_NATIVE_LIB"):
        paths.append(Path(os.environ["VECSTREAM_NATIVE_LIB"]))
    paths.append(Path(__file__).parent / "_native" / _LIB_NAME)
    paths.append(_ROOT / "native" / "target" / "release" / _LIB_NAME)
    return paths


def _bind(lib: ctypes.CDLL) -> None:
    c = ctypes
    f32p, u32p, i32p, i64p = (c.POINTER(x) for x in (c.c_float, c.c_uint32, c.c_int32, c.c_int64))
    sig = {
        "vs_abi_version": (c.c_uint32, []),
        "vs_simd_kind": (c.c_int32, []),
        "vs_force_scalar": (None, [c.c_int32]),
        "vs_new": (c.c_void_p, [c.c_uint32, c.c_uint32, c.c_uint32, c.c_int64, c.c_uint64]),
        "vs_free": (None, [c.c_void_p]),
        "vs_add": (c.c_int64, [c.c_void_p, f32p]),
        "vs_remove": (c.c_int32, [c.c_void_p, c.c_uint32]),
        "vs_len": (c.c_int64, [c.c_void_p]),
        "vs_slots": (c.c_int64, [c.c_void_p]),
        "vs_search": (
            c.c_int64,
            [
                c.c_void_p,
                f32p,
                c.c_uint32,
                c.c_uint32,
                c.c_float,
                _PRED,
                c.c_uint64,
                u32p,
                f32p,
                i32p,
            ],
        ),
        "vs_search_batch": (
            c.c_int32,
            [
                c.c_void_p,
                f32p,
                c.c_uint32,
                c.c_uint32,
                c.c_uint32,
                c.c_float,
                c.c_uint32,
                u32p,
                f32p,
                u32p,
            ],
        ),
        "vs_stats": (c.c_int32, [c.c_void_p, c.POINTER(c.c_double)]),
        "vs_validate": (c.c_int32, [c.c_void_p, c.c_char_p, c.c_uint32]),
        "vs_export_sizes": (c.c_int32, [c.c_void_p, i64p]),
        "vs_export": (c.c_int32, [c.c_void_p, i32p, c.c_uint64, i32p, c.c_uint64]),
        "vs_import": (
            c.c_void_p,
            [c.c_uint32, c.c_uint32, c.c_uint32, c.c_int64, c.c_uint64, c.c_uint64, f32p, i32p]
            + [c.c_uint64, i32p, c.c_int64],
        ),
    }
    for name, (restype, argtypes) in sig.items():
        function = getattr(lib, name)
        function.restype, function.argtypes = restype, argtypes


def load_library() -> Optional[ctypes.CDLL]:
    """Load (once) and return the native library, or None if unavailable."""
    global _library, _load_error
    with _lock:
        if _library is not None or _load_error is not None:
            return _library
        failures = []
        for path in _candidates():
            if not path.exists():
                continue
            try:
                lib = ctypes.CDLL(str(path))
                _bind(lib)
                if lib.vs_abi_version() != ABI_VERSION:
                    raise OSError(f"ABI {lib.vs_abi_version()} != expected {ABI_VERSION}")
                _library = lib
                return lib
            except (OSError, AttributeError) as exc:
                failures.append(f"{path}: {exc}")
        _load_error = (
            "; ".join(failures) or "native library not built (python -m vecstream.native build)"
        )
        return None


def native_available() -> bool:
    return load_library() is not None


def native_status() -> Dict[str, object]:
    lib = load_library()
    return {
        "available": lib is not None,
        "simd": ("avx2+fma" if lib.vs_simd_kind() else "portable") if lib else None,
        "error": None if lib else _load_error,
    }


def build(release: bool = True) -> Path:
    """Compile the Rust crate and copy the library next to the package."""
    profile = ["--release"] if release else []
    subprocess.run(["cargo", "build", *profile], cwd=_ROOT / "native", check=True)
    built = _ROOT / "native" / "target" / ("release" if release else "debug") / _LIB_NAME
    target = Path(__file__).parent / "_native" / _LIB_NAME
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(built.read_bytes())
    return target


def _ptr(array: np.ndarray, ctype):  # type: ignore[no-untyped-def]
    return array.ctypes.data_as(ctypes.POINTER(ctype))


class NativeHNSWIndex:
    """Drop-in replacement for ``HNSWIndex`` backed by the Rust engine.

    ``nodes`` maps item ID to its internal slot (membership, length, and
    iteration match ``HNSWIndex.nodes``; values are slots, not vectors).
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ml: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not (1 <= dim <= MAX_DIM and 2 <= M <= MAX_M and M <= ef_construction <= MAX_EF):
            raise ValueError(
                f"require 1 <= dim <= {MAX_DIM}, 2 <= M <= {MAX_M}, "
                f"M <= ef_construction <= {MAX_EF}"
            )
        lib = load_library()
        if lib is None:
            raise RuntimeError(f"native engine unavailable: {_load_error}")
        self._lib = lib
        self.dim, self.M, self.ef_construction, self.ml = dim, M, ef_construction, ml
        self._seed = int.from_bytes(os.urandom(8), "little") if seed is None else seed
        handle = lib.vs_new(dim, M, ef_construction, -1 if ml is None else ml, self._seed)
        if not handle:
            raise RuntimeError("native engine rejected the index parameters")
        self._attach(handle, {}, [])

    def _attach(self, handle: int, nodes: Dict[str, int], id_of: List[Optional[str]]) -> None:
        self._handle: Optional[int] = handle
        self._lock = ReadWriteLock()  # keeps the ID tables and the native graph in step
        self.nodes = nodes
        self._id_of = id_of

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
    ) -> "NativeHNSWIndex":
        """Restore a persisted graph; slot ``i`` is ``ids[i]``. Raises ``CorruptStoreError``."""
        self = cls(dim, M, ef_construction, ml)
        assert self._handle is not None
        self._lib.vs_free(self._handle)
        self._handle = None
        slab = np.zeros((len(ids), dim), dtype=np.float32)
        for i, item_id in enumerate(ids):
            slab[i] = normalize(vectors[item_id])
        level_array = np.ascontiguousarray(levels, dtype=np.int32)
        edge_array = np.ascontiguousarray(edges, dtype=np.int32).reshape(-1, 3)
        handle = self._lib.vs_import(
            dim,
            M,
            ef_construction,
            -1 if ml is None else ml,
            self._seed,
            len(ids),
            _ptr(slab, ctypes.c_float),
            _ptr(level_array, ctypes.c_int32),
            len(edge_array),
            _ptr(edge_array, ctypes.c_int32),
            -1 if ep is None else ep,
        )
        if not handle:
            raise CorruptStoreError("persisted graph rejected by the native engine")
        self._attach(handle, {item_id: i for i, item_id in enumerate(ids)}, list(ids))
        return self

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        if handle:
            self._lib.vs_free(handle)
            self._handle = None

    def _live(self) -> int:
        if self._handle is None:
            raise RuntimeError("index is closed")
        return self._handle

    def __len__(self) -> int:
        return len(self.nodes)

    def add_item(self, id: str, vector: Sequence[float] | np.ndarray) -> None:
        item_id = str(id)
        value = normalize(as_float32_vector(vector, self.dim))
        with self._lock.write():
            if item_id in self.nodes:
                self._remove_locked(item_id)
            slot = self._lib.vs_add(self._live(), _ptr(value, ctypes.c_float))
            if slot < 0:
                raise RuntimeError(f"native add failed with code {slot}")
            if slot == len(self._id_of):
                self._id_of.append(item_id)
            else:
                self._id_of[slot] = item_id
            self.nodes[item_id] = int(slot)

    def remove_item(self, id: str) -> None:
        with self._lock.write():
            self._remove_locked(id)

    def _remove_locked(self, id: str) -> None:
        if id not in self.nodes:
            raise KeyError(f"Item with ID {id} not found in index")
        slot = self.nodes.pop(id)
        self._id_of[slot] = None
        if self._lib.vs_remove(self._live(), slot) != 1:
            raise RuntimeError("native remove failed")

    def search(
        self,
        query: Sequence[float] | np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
        predicate: Optional[Callable[[str], bool]] = None,
        min_score: Optional[float] = None,
        max_visits: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        if k < 1:
            raise ValueError("k must be at least 1")
        q = normalize(as_float32_vector(query, self.dim))
        with self._lock.read():
            return self._search_locked(q, k, ef_search, predicate, min_score, max_visits)

    def _search_locked(
        self,
        q: np.ndarray,
        k: int,
        ef_search: int | None,
        predicate: Optional[Callable[[str], bool]],
        min_score: Optional[float],
        max_visits: Optional[int],
    ) -> List[Tuple[str, float]]:
        if not self.nodes:
            return []
        slots = np.empty(k, dtype=np.uint32)
        scores = np.empty(k, dtype=np.float32)
        failure: List[BaseException] = []
        callback = _PRED(0)
        if predicate is not None:

            def adapter(slot: int) -> int:
                try:
                    item_id = self._id_of[slot]
                    return int(item_id is not None and bool(predicate(item_id)))
                except BaseException as exc:  # must not unwind through foreign frames
                    failure.append(exc)
                    return 0

            callback = _PRED(adapter)
        count = self._lib.vs_search(
            self._live(),
            _ptr(q, ctypes.c_float),
            k,
            max(k, ef_search or k),
            float("nan") if min_score is None else float(min_score),
            callback,
            max_visits or 0,
            _ptr(slots, ctypes.c_uint32),
            _ptr(scores, ctypes.c_float),
            None,
        )
        if failure:
            raise failure[0]
        if count < 0:
            raise RuntimeError(f"native search failed with code {count}")
        return [(self._id_of[int(slots[i])], float(scores[i])) for i in range(count)]  # type: ignore[misc]

    def search_batch(
        self,
        queries: Sequence[Sequence[float]] | np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
        min_score: Optional[float] = None,
        threads: int = 0,
    ) -> List[List[Tuple[str, float]]]:
        """Search many queries in parallel on native threads (no GIL involved)."""
        if k < 1:
            raise ValueError("k must be at least 1")
        batch = np.stack([normalize(as_float32_vector(q, self.dim)) for q in queries])
        with self._lock.read():
            return self._batch_locked(batch, k, ef_search, min_score, threads)

    def _batch_locked(
        self,
        batch: np.ndarray,
        k: int,
        ef_search: int | None,
        min_score: Optional[float],
        threads: int,
    ) -> List[List[Tuple[str, float]]]:
        n = len(batch)
        if not self.nodes:
            return [[] for _ in range(n)]
        slots = np.empty((n, k), dtype=np.uint32)
        scores = np.empty((n, k), dtype=np.float32)
        counts = np.zeros(n, dtype=np.uint32)
        code = self._lib.vs_search_batch(
            self._live(),
            _ptr(batch, ctypes.c_float),
            n,
            k,
            max(k, ef_search or k),
            float("nan") if min_score is None else float(min_score),
            threads or os.cpu_count() or 1,
            _ptr(slots, ctypes.c_uint32),
            _ptr(scores, ctypes.c_float),
            _ptr(counts, ctypes.c_uint32),
        )
        if code != 0:
            raise RuntimeError(f"native batch search failed with code {code}")
        return [
            [(self._id_of[int(slots[r, j])], float(scores[r, j])) for j in range(int(counts[r]))]  # type: ignore[misc]
            for r in range(n)
        ]

    def validate(self) -> None:
        with self._lock.read():
            self._validate_locked()

    def _validate_locked(self) -> None:
        message = ctypes.create_string_buffer(256)
        code = self._lib.vs_validate(self._live(), message, 256)
        if code != 0:
            raise IndexInvariantError(message.value.decode(errors="replace") or f"code {code}")
        if len(self.nodes) != self._lib.vs_len(self._live()):
            raise IndexInvariantError("ID map and native index disagree on node count")

    def stats(self) -> Dict[str, float]:
        out = (ctypes.c_double * 5)()
        with self._lock.read():
            self._lib.vs_stats(self._live(), out)
        return {
            "nodes": int(out[0]),
            "levels": int(out[1]),
            "avg_degree": float(out[2]),
            "isolated": int(out[3]),
        }

    def export_topology(self, order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        """Levels, directed edges, and entry position with nodes numbered by ``order``."""
        with self._lock.read():
            return self._export_locked(order)

    def _export_locked(self, order: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, Optional[int]]:
        if set(order) != set(self.nodes):
            raise IndexInvariantError("index nodes differ from the vector store")
        sizes = (ctypes.c_int64 * 3)()
        self._lib.vs_export_sizes(self._live(), sizes)
        slots, edge_count, ep = int(sizes[0]), int(sizes[1]), int(sizes[2])
        levels = np.empty(slots, dtype=np.int32)
        edges = np.empty((edge_count, 3), dtype=np.int32)
        code = self._lib.vs_export(
            self._live(),
            _ptr(levels, ctypes.c_int32),
            slots,
            _ptr(edges, ctypes.c_int32),
            edge_count,
        )
        if code != 0:
            raise RuntimeError(f"native export failed with code {code} (index changed mid-export?)")
        position = np.full(slots, -1, dtype=np.int64)
        for i, item_id in enumerate(order):
            position[self.nodes[item_id]] = i
        new_levels = levels[[self.nodes[i] for i in order]] if len(order) else levels[:0]
        if edge_count:
            edges = np.stack(
                [edges[:, 0], position[edges[:, 1]], position[edges[:, 2]]], axis=1
            ).astype(np.int32)
        return new_levels, edges, (int(position[ep]) if ep >= 0 else None)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        print(f"built {build()}")
    else:
        print(native_status())
