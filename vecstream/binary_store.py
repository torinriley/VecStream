"""Versioned, crash-safe local persistence for vectors and metadata."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import CorruptStoreError, InvalidManifestError, UnsupportedFormatVersionError
from .vector_store import VectorStore

FORMAT_VERSION = 1


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())


class BinaryVectorStore(VectorStore):
    """A VectorStore checkpointed as immutable generations.

    ``CURRENT`` is the only overwritten file. It is atomically replaced after
    every component of a new generation has been flushed. It records the prior
    generation, allowing recovery if the newest generation is later damaged.
    """

    def __init__(self, storage_dir: str) -> None:
        super().__init__()
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
            raise CorruptStoreError(f"cannot read checkpoint pointer {self.current_file}: {exc}") from exc
        candidates = [pointer.get("current"), pointer.get("previous")]
        failures = []
        for generation in filter(None, candidates):
            try:
                self._load_generation(str(generation))
                return
            except CorruptStoreError as exc:
                failures.append(f"{generation}: {exc}")
        raise CorruptStoreError("no valid checkpoint generation; " + "; ".join(failures))

    def _load_generation(self, generation: str) -> None:
        directory = self.root / generation
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidManifestError(f"manifest is unreadable: {exc}") from exc
        if manifest.get("format_version") != FORMAT_VERSION:
            raise UnsupportedFormatVersionError(
                f"format version {manifest.get('format_version')!r} is not supported"
            )
        required = {"dimension", "dtype", "metric", "vector_count"}
        if not required.issubset(manifest) or manifest["dtype"] != "float32" or manifest["metric"] != "cosine":
            raise InvalidManifestError("manifest fields, dtype, or metric are invalid")
        try:
            ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            vectors = np.load(directory / "vectors.npy", allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CorruptStoreError(f"checkpoint component is unreadable: {exc}") from exc
        expected_shape = (manifest["vector_count"], manifest["dimension"])
        if vectors.dtype != np.float32 or vectors.shape != expected_shape or len(ids) != manifest["vector_count"]:
            raise CorruptStoreError(
                f"vector checkpoint mismatch: expected {expected_shape} float32, got {vectors.shape} {vectors.dtype}"
            )
        if len(ids) != len(set(ids)) or not isinstance(metadata, dict):
            raise CorruptStoreError("IDs must be unique and metadata must be an object")
        if not np.all(np.isfinite(vectors)):
            raise CorruptStoreError("persisted vectors contain NaN or infinity")
        self.vectors = {str(item_id): vectors[i].copy() for i, item_id in enumerate(ids)}
        self.metadata = metadata
        self.dimension = int(manifest["dimension"]) if ids else None
        self._generation = generation

    def _save_store(self) -> None:
        generation = f"gen-{uuid.uuid4().hex}"
        directory = self.root / generation
        directory.mkdir()
        ids = list(self.vectors)
        dimension = self.dimension or 0
        matrix = (np.stack([self.vectors[item_id] for item_id in ids]).astype(np.float32)
                  if ids else np.empty((0, dimension), dtype=np.float32))
        try:
            vectors_path = directory / "vectors.npy"
            with vectors_path.open("wb") as handle:
                np.save(handle, matrix, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            _write_json(directory / "ids.json", ids)
            _write_json(directory / "metadata.json", self.metadata)
            _write_json(directory / "manifest.json", {
                "format_version": FORMAT_VERSION,
                "dimension": dimension,
                "dtype": "float32",
                "metric": "cosine",
                "vector_count": len(ids),
            })
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
            temporary = self.root / f".CURRENT-{uuid.uuid4().hex}"
            _write_json(temporary, {"current": generation, "previous": self._generation})
            os.replace(temporary, self.current_file)
            self._generation = generation
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def add_vector(self, id: str, vector: Sequence[float] | np.ndarray,
                   metadata: Optional[dict] = None) -> None:
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

    def get_vector_with_metadata(self, id: str) -> Tuple[List[float], Optional[dict]]:
        return self.get_vector(id), self.metadata.get(id)

    def clear_store(self) -> None:
        self.vectors, self.metadata, self.dimension = {}, {}, None
        self._save_store()

    def get_store_size(self) -> Tuple[int, int]:
        if self._generation is None:
            return 0, 0
        directory = self.root / self._generation
        return (directory.joinpath("vectors.npy").stat().st_size,
                directory.joinpath("metadata.json").stat().st_size)
