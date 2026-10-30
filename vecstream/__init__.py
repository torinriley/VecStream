"""VecStream: compact exact and HNSW vector search."""

__version__ = "0.4.0"

from .auth import KeyStore, Principal
from .binary_store import BinaryVectorStore
from .collections import Collection, CollectionManager
from .errors import (
    AuthenticationError,
    AuthorizationError,
    ClosedError,
    CorruptStoreError,
    DimensionMismatchError,
    IndexInvariantError,
    InvalidManifestError,
    InvalidVectorError,
    StoreLockedError,
    UnsupportedFormatVersionError,
    WALCorruptionError,
)
from .hnsw_index import HNSWIndex
from .server import VecStreamServer
from .service import VecStreamService
from .vector_store import VectorStore

__all__ = [
    "AuthenticationError",
    "AuthorizationError",
    "ClosedError",
    "KeyStore",
    "Principal",
    "StoreLockedError",
    "VecStreamServer",
    "VecStreamService",
    "WALCorruptionError",
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
