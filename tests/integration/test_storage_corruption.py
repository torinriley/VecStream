import json

import pytest

from vecstream.binary_store import BinaryVectorStore
from vecstream.errors import CorruptStoreError


def test_round_trip_uses_non_pickle_matrix(tmp_path):
    store = BinaryVectorStore(str(tmp_path))
    store.add_vector("a", [1, 2, 3], {"kind": "test"})
    loaded = BinaryVectorStore(str(tmp_path))
    assert loaded.get_vector_with_metadata("a") == ([1.0, 2.0, 3.0], {"kind": "test"})


def test_recovers_previous_generation_when_current_is_truncated(tmp_path):
    store = BinaryVectorStore(str(tmp_path))
    store.add_vector("a", [1, 0])
    store.add_vector("b", [0, 1])
    pointer = json.loads((tmp_path / "CURRENT").read_text())
    (tmp_path / pointer["current"] / "vectors.npy").write_bytes(b"truncated")
    recovered = BinaryVectorStore(str(tmp_path))
    assert list(recovered.vectors) == ["a"]


def test_corruption_never_becomes_an_empty_store(tmp_path):
    store = BinaryVectorStore(str(tmp_path))
    store.add_vector("a", [1, 0])
    pointer = json.loads((tmp_path / "CURRENT").read_text())
    (tmp_path / pointer["current"] / "manifest.json").write_text("not-json")
    with pytest.raises(CorruptStoreError):
        BinaryVectorStore(str(tmp_path))
