"""
VecStream - A lightweight, efficient vector database with similarity search capabilities.
"""

__version__ = "0.1.1"
__author__ = "Torin Etheridge"

from .vector_store import VectorStore
from .persistent_store import PersistentVectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine

__all__ = [
    'VectorStore',
    'PersistentVectorStore',
    'IndexManager',
    'QueryEngine',
]

