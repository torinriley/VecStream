"""Golden bytes: fail loudly if an on-disk format changes without a version bump."""

import json
import struct
import zlib

import numpy as np

from vecstream import BinaryVectorStore
from vecstream.wal import MAGIC, WriteAheadLog


def test_wal_frame_layout_is_stable(tmp_path):
    wal = WriteAheadLog(tmp_path / "wal.log", sync=False)
    wal.append(b"hello")
    wal.close()
    data = (tmp_path / "wal.log").read_bytes()
    header = struct.pack("<IQ", 5, 1)
    expected = (
        MAGIC
        + header
        + struct.pack("<I", zlib.crc32(header))
        + b"hello"
        + struct.pack("<I", zlib.crc32(b"hello", zlib.crc32(struct.pack("<Q", 1))))
    )
    assert MAGIC == b"VSWAL001" and data == expected


def test_checkpoint_manifest_schema_is_stable(tmp_path):
    store = BinaryVectorStore(str(tmp_path))
    store.add_vector("a", [1.0, 2.0], {"k": 1})
    root = tmp_path
    pointer = json.loads((root / "CURRENT").read_text())
    assert set(pointer) == {"current", "previous", "lsn"}
    manifest = json.loads((root / pointer["current"] / "manifest.json").read_text())
    assert manifest["format_version"] == 2
    assert set(manifest) == {
        "format_version",
        "dimension",
        "dtype",
        "metric",
        "vector_count",
        "lsn",
        "checksums",
        "extra_files",
    }
    assert set(manifest["checksums"]) == {"vectors.npy", "ids.json", "metadata.json"}
    assert np.load(root / pointer["current"] / "vectors.npy").dtype == np.float32


def test_version_1_checkpoints_without_checksums_remain_readable(tmp_path):
    generation = tmp_path / "gen-legacy"
    generation.mkdir()
    np.save(generation / "vectors.npy", np.asarray([[1.0, 0.0]], dtype=np.float32))
    (generation / "ids.json").write_text('["a"]')
    (generation / "metadata.json").write_text('{"a": {"x": 1}}')
    (generation / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "dimension": 2,
                "dtype": "float32",
                "metric": "cosine",
                "vector_count": 1,
            }
        )
    )
    (tmp_path / "CURRENT").write_text(json.dumps({"current": "gen-legacy", "previous": None}))
    store = BinaryVectorStore(str(tmp_path))
    assert store.get_vector_with_metadata("a") == ([1.0, 0.0], {"x": 1})
    store.add_vector("b", [0.0, 1.0])  # upgrades to v2 on next write
    assert BinaryVectorStore(str(tmp_path)).vectors.keys() == {"a", "b"}
