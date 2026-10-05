"""VecStream: compact exact and HNSW vector search."""

from .binary_store import BinaryVectorStore
from .collections import Collection, CollectionManager
from .errors import (
    CorruptStoreError,
    DimensionMismatchError,
    IndexInvariantError,
    InvalidManifestError,
    InvalidVectorError,
    UnsupportedFormatVersionError,
)
from .hnsw_index import HNSWIndex
from .vector_store import VectorStore

__all__ = [
    "BinaryVectorStore",
    "Collection",
    "CollectionManager",
    "CorruptStoreError",
    "DimensionMismatchError",
    "HNSWIndex",
    "IndexInvariantError",
    "InvalidManifestError",
    "InvalidVectorError",
    "UnsupportedFormatVersionError",
    "VectorStore",
]
