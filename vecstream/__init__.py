"""
VecStream - a compact vector-search engine.
"""

__version__ = "0.4.0"

from .vector_store import VectorStore
from .binary_store import BinaryVectorStore
from .persistent_store import PersistentVectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine
from .hnsw_index import HNSWIndex
from .collections import Collection, CollectionManager
from .client import ClientAPI
from .errors import (
    CorruptStoreError,
    DimensionMismatchError,
    IndexInvariantError,
    InvalidManifestError,
    InvalidVectorError,
    UnsupportedFormatVersionError,
)

__all__ = [
    "VectorStore",
    "BinaryVectorStore",
    "PersistentVectorStore",
    "IndexManager",
    "QueryEngine",
    "HNSWIndex",
    "Collection",
    "CollectionManager",
    "ClientAPI",
    "CorruptStoreError",
    "DimensionMismatchError",
    "IndexInvariantError",
    "InvalidManifestError",
    "InvalidVectorError",
    "UnsupportedFormatVersionError",
]
