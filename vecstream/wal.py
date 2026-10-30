"""Checksummed write-ahead log.

File layout: ``MAGIC`` followed by frames. Each frame is::

    <u32 payload_len><u64 lsn><u32 header_crc> payload <u32 payload_crc>

``header_crc`` covers ``payload_len`` and ``lsn``; ``payload_crc`` covers
``lsn`` and the payload. A frame is committed once all of its bytes are
durable, so a torn final frame is discarded and never partially applied.
Damage anywhere else is reported rather than silently dropped.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path
from typing import List, Tuple

from .errors import WALCorruptionError

MAGIC = b"VSWAL001"
_HEADER = struct.Struct("<IQI")
_TRAILER = struct.Struct("<I")
MAX_PAYLOAD = 1 << 30


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _header_crc(length: int, lsn: int) -> int:
    return zlib.crc32(struct.pack("<IQ", length, lsn))


def _payload_crc(lsn: int, payload: bytes) -> int:
    return zlib.crc32(payload, zlib.crc32(struct.pack("<Q", lsn)))


def _scan(
    data: bytes, after_lsn: int, truncate_on_damage: bool
) -> Tuple[List[Tuple[int, bytes]], int, int]:
    """Return (records with lsn > after_lsn, valid_end_offset, last_lsn)."""
    if len(data) < len(MAGIC):
        return [], 0, after_lsn
    if data[: len(MAGIC)] != MAGIC:
        raise WALCorruptionError("write-ahead log has an invalid header")
    offset = len(MAGIC)
    records: List[Tuple[int, bytes]] = []
    last_lsn = after_lsn
    previous: int | None = None
    while offset < len(data):
        remaining = len(data) - offset
        if remaining < _HEADER.size:
            break  # torn header at EOF
        length, lsn, header_crc = _HEADER.unpack_from(data, offset)
        if header_crc != _header_crc(length, lsn) or length > MAX_PAYLOAD:
            if truncate_on_damage:
                break
            raise WALCorruptionError(f"write-ahead log frame header damaged at offset {offset}")
        end = offset + _HEADER.size + length + _TRAILER.size
        if end > len(data):
            break  # torn payload at EOF
        payload = data[offset + _HEADER.size : end - _TRAILER.size]
        (payload_crc,) = _TRAILER.unpack_from(data, end - _TRAILER.size)
        if payload_crc != _payload_crc(lsn, payload):
            if end == len(data) or truncate_on_damage:
                break  # torn / unsyncable final frame
            raise WALCorruptionError(f"write-ahead log checksum mismatch at offset {offset}")
        if previous is not None and lsn != previous + 1:
            if truncate_on_damage:
                break
            raise WALCorruptionError(f"write-ahead log LSN gap: {previous} -> {lsn}")
        previous = lsn
        if lsn > after_lsn:
            records.append((lsn, payload))
            last_lsn = lsn
        offset = end
    return records, offset, last_lsn


class WriteAheadLog:
    """Append-only durable log of opaque payloads."""

    def __init__(
        self, path: Path, checkpoint_lsn: int = 0, sync: bool = True, on_corruption: str = "raise"
    ) -> None:
        if on_corruption not in ("raise", "truncate"):
            raise ValueError("on_corruption must be 'raise' or 'truncate'")
        self.path = Path(path)
        self.sync = sync
        self.checkpoint_lsn = checkpoint_lsn
        self.recovered_records: List[Tuple[int, bytes]] = []
        self.truncated_bytes = 0
        data = self.path.read_bytes() if self.path.exists() else b""
        records, valid_end, last_lsn = _scan(data, checkpoint_lsn, on_corruption == "truncate")
        self.recovered_records = records
        self.last_lsn = max(last_lsn, checkpoint_lsn)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if valid_end == 0:
                os.ftruncate(self._fd, 0)
                os.pwrite(self._fd, MAGIC, 0)
                valid_end = len(MAGIC)
                self.truncated_bytes = len(data)
            elif valid_end < len(data):
                self.truncated_bytes = len(data) - valid_end
                os.ftruncate(self._fd, valid_end)
            if self.sync:
                os.fsync(self._fd)
            os.lseek(self._fd, valid_end, os.SEEK_SET)
        except BaseException:
            os.close(self._fd)
            raise
        self.size = valid_end

    def append(self, payload: bytes) -> int:
        """Durably append ``payload`` and return its LSN."""
        lsn = self.last_lsn + 1
        frame = (
            _HEADER.pack(len(payload), lsn, _header_crc(len(payload), lsn))
            + payload
            + _TRAILER.pack(_payload_crc(lsn, payload))
        )
        try:
            written = os.write(self._fd, frame)
            if written != len(frame):
                raise OSError("short write to write-ahead log")
            if self.sync:
                os.fsync(self._fd)
        except BaseException:
            try:  # drop any partial frame so later appends stay well-formed
                os.ftruncate(self._fd, self.size)
                os.lseek(self._fd, self.size, os.SEEK_SET)
            except OSError:
                pass
            raise
        self.size += len(frame)
        self.last_lsn = lsn
        return lsn

    def reset(self, checkpoint_lsn: int) -> None:
        """Atomically replace the log with an empty one after a checkpoint."""
        temporary = self.path.with_name(self.path.name + ".new")
        with temporary.open("wb") as handle:
            handle.write(MAGIC)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        fsync_directory(self.path.parent)
        os.close(self._fd)
        self._fd = os.open(self.path, os.O_RDWR)
        os.lseek(self._fd, len(MAGIC), os.SEEK_SET)
        self.size = len(MAGIC)
        self.checkpoint_lsn = checkpoint_lsn
        self.last_lsn = max(self.last_lsn, checkpoint_lsn)

    def close(self) -> None:
        if self._fd >= 0:
            try:
                if self.sync:
                    os.fsync(self._fd)
            finally:
                os.close(self._fd)
                self._fd = -1
