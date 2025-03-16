"""
Common test fixtures and configuration for VecStream tests.
"""
import os
import pytest
import tempfile
from vecstream import VectorStore, PersistentVectorStore

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
def sample_vectors():
    """Fixture that provides sample vectors for testing."""
    return [
        ("vec1", [1.0, 0.0, 0.0]),
        ("vec2", [0.0, 1.0, 0.0]),
        ("vec3", [0.0, 0.0, 1.0]),
    ] 