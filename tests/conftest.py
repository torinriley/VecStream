"""
Common test fixtures and configuration for VecStream tests.
"""
import os
import pytest
import tempfile
import numpy as np
from vecstream import (
    VectorStore, 
    PersistentVectorStore,
    BinaryVectorStore,
    HNSWIndex,
    Collection,
    CollectionManager
)

@pytest.fixture
def vector_store():
    """Fixture that provides a clean VectorStore instance."""
    return VectorStore()

@pytest.fixture
def persistent_store():
    """Fixture that provides a temporary PersistentVectorStore instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = PersistentVectorStore(os.path.join(temp_dir, "test.vec"))
        yield store

@pytest.fixture
def binary_store():
    """Fixture that provides a temporary BinaryVectorStore instance."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = BinaryVectorStore(temp_dir)
        yield store

@pytest.fixture
def sample_vectors():
    """Fixture that provides sample vectors for testing."""
    return [
        ("vec1", [1.0, 0.0, 0.0]),
        ("vec2", [0.0, 1.0, 0.0]),
        ("vec3", [0.0, 0.0, 1.0]),
    ]

@pytest.fixture
def hnsw_index():
    """Fixture that provides a clean HNSW index."""
    return HNSWIndex(dim=3, M=16, ef_construction=100)

@pytest.fixture
def populated_hnsw_index(hnsw_index, sample_vectors):
    """Fixture that provides an HNSW index populated with sample vectors."""
    for id, vector in sample_vectors:
        hnsw_index.add_item(id, np.array(vector))
    return hnsw_index

@pytest.fixture
def temp_directory():
    """Fixture that provides a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def collection_manager(temp_directory):
    """Fixture that provides a CollectionManager instance."""
    return CollectionManager(temp_directory)

@pytest.fixture
def test_collection(collection_manager):
    """Fixture that provides a test collection."""
    collection = collection_manager.create_collection("test")
    return collection 