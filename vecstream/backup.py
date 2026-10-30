"""Verifiable backups of a collection: a self-contained copy of one checkpoint generation.

A backup is a directory holding the generation's files (``manifest.json``, vectors, IDs,
metadata, and the persisted graph) plus ``backup.json``, which records the source collection,
its log sequence number, and its HNSW parameters, and pins the manifest by SHA-256. Every
component is checksummed by the manifest, so ``verify_backup`` detects any later damage.
Backups are published atomically: a directory either exists complete or does not exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .binary_store import _CORE_FILES, READABLE_FORMAT_VERSIONS, BinaryVectorStore
from .errors import CorruptStoreError, InvalidManifestError, UnsupportedFormatVersionError
from .wal import fsync_directory

BACKUP_FORMAT = 1
INFO_FILE = "backup.json"
MANIFEST_FILE = "manifest.json"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def validate_backup_id(backup_id: str) -> str:
    """Backup identifiers are single path segments; anything else could escape the backup root."""
    if not isinstance(backup_id, str) or not ID_PATTERN.match(backup_id) or backup_id.endswith("."):
        raise ValueError(
            "backup id must be 1-128 characters of letters, digits, '_', '-', '.' "
            "and start with a letter or digit"
        )
    return backup_id


def default_backup_id(collection: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{collection}-{stamp}"[:128]


def stage_generation(generation: Path, stage: Path) -> None:
    """Link (or, across filesystems, copy) an immutable generation's files into ``stage``."""
    stage.mkdir()
    for entry in sorted(generation.iterdir()):
        if not entry.is_file() or entry.is_symlink():
            continue
        try:
            os.link(entry, stage / entry.name)
        except OSError:
            shutil.copyfile(entry, stage / entry.name)


def _copy_durably(source: Path, target: Path) -> None:
    with open(source, "rb") as reader, open(target, "xb") as writer:
        shutil.copyfileobj(reader, writer, 1 << 20)
        writer.flush()
        os.fsync(writer.fileno())


def finalize_backup(stage: Path, destination: Path, info: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy a staged generation to ``destination``, verify it, and publish it atomically."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"backup destination {str(destination)!r} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    partial.mkdir()
    try:
        for entry in sorted(stage.iterdir()):
            _copy_durably(entry, partial / entry.name)
        manifest_bytes = (partial / MANIFEST_FILE).read_bytes()
        record = {
            **info,
            "backup_format": BACKUP_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
        (partial / INFO_FILE).write_text(json.dumps(record, indent=2, sort_keys=True))
        with open(partial / INFO_FILE, "rb") as handle:
            os.fsync(handle.fileno())
        verified = verify_backup(partial)  # a backup that cannot be read back is not a backup
        fsync_directory(partial)
        if destination.exists():
            raise FileExistsError(f"backup destination {str(destination)!r} already exists")
        os.rename(partial, destination)
        fsync_directory(destination.parent)
        return verified
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def verify_backup(path: Path) -> Dict[str, Any]:
    """Return the backup's info after checking its format and every component checksum.

    Raises ``CorruptStoreError`` (or a subclass) if anything is missing, altered, or unsupported.
    """
    path = Path(path)
    try:
        info = json.loads((path / INFO_FILE).read_text(encoding="utf-8"))
        manifest_bytes = (path / MANIFEST_FILE).read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as exc:
        raise InvalidManifestError(f"backup {str(path)!r} is unreadable: {exc}") from exc
    if not isinstance(info, dict) or info.get("backup_format") != BACKUP_FORMAT:
        raise UnsupportedFormatVersionError(
            f"backup format {info.get('backup_format') if isinstance(info, dict) else None!r} "
            "is not supported"
        )
    if info.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise CorruptStoreError("backup manifest does not match the recorded digest")
    if not isinstance(manifest, dict) or manifest.get("format_version") not in (
        READABLE_FORMAT_VERSIONS
    ):
        raise UnsupportedFormatVersionError("backup checkpoint format is not supported")
    for name in {INFO_FILE, MANIFEST_FILE, *_CORE_FILES, *manifest.get("extra_files", [])}:
        member = path / Path(name).name
        if member.is_symlink() or not member.is_file():
            raise CorruptStoreError(f"backup member {name!r} is missing or not a regular file")
    BinaryVectorStore._read_components(path, manifest)  # checksums of every component
    return info


def install_backup(source: Path, collection_dir: Path, lsn: int) -> str:
    """Lay a verified backup out as the sole checkpoint generation of ``collection_dir``."""
    generation = f"gen-{uuid.uuid4().hex}"
    target = collection_dir / generation
    target.mkdir(parents=True)
    for entry in sorted(Path(source).iterdir()):
        if entry.name == INFO_FILE:
            continue
        _copy_durably(entry, target / entry.name)
    fsync_directory(target)
    pointer = collection_dir / "CURRENT"
    pointer.write_text(json.dumps({"current": generation, "previous": None, "lsn": lsn}))
    with open(pointer, "rb") as handle:
        os.fsync(handle.fileno())
    fsync_directory(collection_dir)
    return generation


def list_backups(root: Path) -> List[Dict[str, Any]]:
    """Every readable-looking backup directly under ``root`` (not verified; see verify_backup)."""
    found: List[Dict[str, Any]] = []
    if not root.is_dir():
        return found
    for entry in sorted(root.iterdir()):
        if entry.name.startswith(".") or not (entry / INFO_FILE).is_file():
            continue
        try:
            info = json.loads((entry / INFO_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(info, dict):
            found.append({"id": entry.name, **info})
    return found
