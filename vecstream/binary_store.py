"""Versioned, crash-safe local persistence for vectors and metadata."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .errors import CorruptStoreError, InvalidManifestError, UnsupportedFormatVersionError
from .vector_store import VectorStore
from .wal import fsync_directory

FORMAT_VERSION = 2
READABLE_FORMAT_VERSIONS = (1, 2)
_CORE_FILES = ("vectors.npy", "ids.json", "metadata.json")


def _write_bytes(path: Path, data: bytes) -> str:
    """Write and fsync ``data``; return its SHA-256 hex digest."""
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class BinaryVectorStore(VectorStore):
    """A VectorStore checkpointed as immutable generations.

    ``CURRENT`` is the only overwritten file. It is atomically replaced after
    every component of a new generation has been flushed. It records the prior
    generation, allowing recovery if the newest generation is later damaged.
    Every component carries a SHA-256 checksum in the manifest that is verified
    on load; only the current and previous generations are retained.

    With ``autosave=False`` mutations stay in memory until ``save_checkpoint``;
    a write-ahead log owned by the caller provides their durability.
    """

    def __init__(self, storage_dir: str, autosave: bool = True) -> None:
        super().__init__()
        self.autosave = autosave
        self.checkpoint_lsn = 0
        self.pointer_lsn: Optional[int] = None
        self.fell_back = False
        self.extra_files: Dict[str, bytes] = {}
        self.storage_dir = str(storage_dir)
        self.root = Path(storage_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.current_file = self.root / "CURRENT"
        self.metadata: Dict[str, dict] = {}
        self._generation: Optional[str] = None
        self._load_store()

    def _load_store(self) -> None:
        if not self.current_file.exists():
            return
        try:
            pointer = json.loads(self.current_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CorruptStoreError(
                f"cannot read checkpoint pointer {self.current_file}: {exc}"
            ) from exc
        candidates = [pointer.get("current"), pointer.get("previous")]
        self.pointer_lsn = pointer.get("lsn")
        failures = []
        for generation in filter(None, candidates):
            try:
                self._load_generation(str(generation))
                self.fell_back = generation != candidates[0]
                return
            except CorruptStoreError as exc:
                failures.append(f"{generation}: {exc}")
        raise CorruptStoreError("no valid checkpoint generation; " + "; ".join(failures))

    @staticmethod
    def _read_components(directory: Path, manifest: dict) -> Dict[str, bytes]:
        """Read every component, verifying checksums recorded by format v2."""
        names = list(_CORE_FILES) + list(manifest.get("extra_files", []))
        checksums = manifest.get("checksums")
        if manifest.get("format_version", 1) >= 2 and not isinstance(checksums, dict):
            raise InvalidManifestError("manifest is missing component checksums")
        components: Dict[str, bytes] = {}
        for name in names:
            if Path(name).name != name:
                raise InvalidManifestError(f"invalid component name {name!r}")
            try:
                data = (directory / name).read_bytes()
            except OSError as exc:
                raise CorruptStoreError(f"component {name} is unreadable: {exc}") from exc
            if checksums is not None and checksums.get(name) != hashlib.sha256(data).hexdigest():
                raise CorruptStoreError(f"checksum mismatch for checkpoint component {name}")
            components[name] = data
        return components

    def _load_generation(self, generation: str) -> None:
        directory = self.root / generation
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidManifestError(f"manifest is unreadable: {exc}") from exc
        if not isinstance(manifest, dict) or manifest.get("format_version") not in (
            READABLE_FORMAT_VERSIONS
        ):
            raise UnsupportedFormatVersionError(
                f"format version {manifest.get('format_version')!r} is not supported"
            )
        required = {"dimension", "dtype", "metric", "vector_count"}
        if not isinstance(manifest, dict) or (
            not required.issubset(manifest)
            or manifest["dtype"] != "float32"
            or manifest["metric"] != "cosine"
        ):
            raise InvalidManifestError("manifest fields, dtype, or metric are invalid")
        components = self._read_components(directory, manifest)
        try:
            ids = json.loads(components["ids.json"])
            metadata = json.loads(components["metadata.json"])
            vectors = np.load(io.BytesIO(components["vectors.npy"]), allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CorruptStoreError(f"checkpoint component is unreadable: {exc}") from exc
        if not isinstance(ids, list):
            raise CorruptStoreError("IDs must be a list")
        expected_shape = (manifest["vector_count"], manifest["dimension"])
        if (
            vectors.dtype != np.float32
            or vectors.shape != expected_shape
            or len(ids) != manifest["vector_count"]
        ):
            raise CorruptStoreError(
                "vector checkpoint mismatch: "
                f"expected {expected_shape} float32, got {vectors.shape} {vectors.dtype}"
            )
        if len(ids) != len(set(ids)) or not isinstance(metadata, dict):
            raise CorruptStoreError("IDs must be unique and metadata must be an object")
        if not np.all(np.isfinite(vectors)):
            raise CorruptStoreError("persisted vectors contain NaN or infinity")
        self.extra_files = {
            name: data for name, data in components.items() if name not in _CORE_FILES
        }
        self.checkpoint_lsn = int(manifest.get("lsn", 0))
        self.vectors = {str(item_id): vectors[i].copy() for i, item_id in enumerate(ids)}
        self.metadata = metadata
        self.dimension = int(manifest["dimension"]) if ids else None
        self._generation = generation

    def save_checkpoint(
        self, extra_files: Optional[Mapping[str, bytes]] = None, lsn: int = 0
    ) -> None:
        """Write a new immutable generation and atomically make it current."""
        generation = f"gen-{uuid.uuid4().hex}"
        directory = self.root / generation
        directory.mkdir()
        ids = list(self.vectors)
        dimension = self.dimension or 0
        matrix = (
            np.stack([self.vectors[item_id] for item_id in ids]).astype(np.float32)
            if ids
            else np.empty((0, dimension), dtype=np.float32)
        )
        extras = dict(extra_files or {})
        try:
            buffer = io.BytesIO()
            np.save(buffer, matrix, allow_pickle=False)
            files: Dict[str, bytes] = {
                "vectors.npy": buffer.getvalue(),
                "ids.json": _json_bytes(ids),
                "metadata.json": _json_bytes(self.metadata),
                **extras,
            }
            checksums = {name: _write_bytes(directory / name, data) for name, data in files.items()}
            _write_bytes(
                directory / "manifest.json",
                _json_bytes(
                    {
                        "format_version": FORMAT_VERSION,
                        "dimension": dimension,
                        "dtype": "float32",
                        "metric": "cosine",
                        "vector_count": len(ids),
                        "lsn": lsn,
                        "checksums": checksums,
                        "extra_files": sorted(extras),
                    }
                ),
            )
            fsync_directory(directory)
            temporary = self.root / f".CURRENT-{uuid.uuid4().hex}"
            _write_bytes(
                temporary,
                _json_bytes({"current": generation, "previous": self._generation, "lsn": lsn}),
            )
            os.replace(temporary, self.current_file)
            fsync_directory(self.root)
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        previous = self._generation
        self._generation = generation
        self.checkpoint_lsn = lsn
        self.pointer_lsn = lsn
        self.fell_back = False
        self.extra_files = extras
        self._collect_garbage(keep={generation, previous})

    def _save_store(self) -> None:
        if self.autosave:
            self.save_checkpoint()

    def _collect_garbage(self, keep: set) -> None:
        """Drop superseded generations and debris from interrupted saves."""
        for entry in self.root.iterdir():
            stale_generation = entry.name.startswith("gen-") and entry.name not in keep
            if stale_generation and entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            elif entry.name.startswith(".CURRENT-") and entry.is_file():
                entry.unlink(missing_ok=True)

    def verify(self) -> None:
        """Re-read and checksum the current generation; raise if damaged."""
        if self._generation is None:
            return
        directory = self.root / self._generation
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidManifestError(f"manifest is unreadable: {exc}") from exc
        self._read_components(directory, manifest)

    def add_vector(
        self, id: str, vector: Sequence[float] | np.ndarray, metadata: Optional[dict] = None
    ) -> None:
        old_vector = self.vectors.get(id)
        old_metadata = self.metadata.get(id)
        super().add_vector(id, vector)
        if metadata is not None:
            self.metadata[id] = metadata
        try:
            self._save_store()
        except BaseException:
            if old_vector is None:
                self.vectors.pop(id, None)
                self.metadata.pop(id, None)
            else:
                self.vectors[id] = old_vector
                if old_metadata is None:
                    self.metadata.pop(id, None)
                else:
                    self.metadata[id] = old_metadata
            raise

    def remove_vector(self, id: str) -> None:
        old_vector = self.vectors.get(id)
        old_metadata = self.metadata.get(id)
        super().remove_vector(id)
        self.metadata.pop(id, None)
        try:
            self._save_store()
        except BaseException:
            assert old_vector is not None
            self.vectors[id] = old_vector
            self.dimension = int(old_vector.size)
            if old_metadata is not None:
                self.metadata[id] = old_metadata
            raise

    def put_in_memory(
        self, id: str, vector: Sequence[float] | np.ndarray, metadata: Optional[dict]
    ) -> None:
        """Upsert a vector and replace its metadata without checkpointing."""
        VectorStore.add_vector(self, id, vector)
        if metadata is None:
            self.metadata.pop(id, None)
        else:
            self.metadata[id] = metadata

    def delete_in_memory(self, id: str) -> None:
        VectorStore.remove_vector(self, id)
        self.metadata.pop(id, None)

    def get_vector_with_metadata(self, id: str) -> Tuple[List[float], Optional[dict]]:
        return self.get_vector(id), self.metadata.get(id)

    def clear_store(self) -> None:
        self.vectors, self.metadata, self.dimension = {}, {}, None
        self._save_store()

    def get_store_size(self) -> Tuple[int, int]:
        if self._generation is None:
            return 0, 0
        directory = self.root / self._generation
        return (
            directory.joinpath("vectors.npy").stat().st_size,
            directory.joinpath("metadata.json").stat().st_size,
        )
