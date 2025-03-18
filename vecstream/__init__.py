"""
VecStream - A lightweight vector database with similarity search
"""

__version__ = "0.3.2"

from .vector_store import VectorStore
from .binary_store import BinaryVectorStore
from .persistent_store import PersistentVectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine
from .hnsw_index import HNSWIndex
from .collections import Collection, CollectionManager

__all__ = [
    "VectorStore",
    "BinaryVectorStore",
    "PersistentVectorStore",
    "IndexManager", 
    "QueryEngine",
    "HNSWIndex",
    "Collection",
    "CollectionManager"
]

