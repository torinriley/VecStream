"""
Unit tests for the Collections and CollectionManager classes.
"""
import os
import pytest
import tempfile
import shutil
import numpy as np
from vecstream import Collection, CollectionManager

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def collection_manager(temp_dir):
    """Fixture that provides a CollectionManager with a temporary directory."""
    return CollectionManager(temp_dir, use_hnsw=True)

@pytest.fixture
def sample_collection(collection_manager):
    """Fixture that provides a sample collection with vectors."""
    collection = collection_manager.create_collection("test_collection")
    
    # Add sample vectors with metadata
    collection.add_vector("vec1", [1.0, 0.0, 0.0], {"tag": "red", "score": 10})
    collection.add_vector("vec2", [0.0, 1.0, 0.0], {"tag": "green", "score": 20})
    collection.add_vector("vec3", [0.0, 0.0, 1.0], {"tag": "blue", "score": 30})
    collection.add_vector("vec4", [0.7, 0.7, 0.0], {"tag": "yellow", "score": 25})
    
    return collection

def test_collection_initialization(temp_dir):
    """Test that Collection initializes correctly."""
    collection = Collection("test", temp_dir)
    
    assert collection.name == "test"
    assert collection.collection_dir == os.path.join(temp_dir, "collections", "test")
    assert collection.use_hnsw is True
    assert collection.hnsw_index is None  # Should be None until vectors are added
    assert os.path.exists(collection.collection_dir)

def test_collection_add_vector(sample_collection):
    """Test adding vectors to a collection."""
    # Test existing vectors
    assert len(sample_collection.store.vectors) == 4
    
    # Add a new vector
    sample_collection.add_vector("vec5", [0.5, 0.5, 0.5], {"tag": "gray", "score": 15})
    
    # Verify it was added to the store
    assert "vec5" in sample_collection.store.vectors
    assert sample_collection.store.dimension == 3
    
    # Get metadata and verify
    _, metadata = sample_collection.get_vector_with_metadata("vec5")
    assert metadata["tag"] == "gray"
    assert metadata["score"] == 15
    
    # Verify it was added to the HNSW index if enabled
    if sample_collection.use_hnsw and sample_collection.hnsw_index:
        assert "vec5" in sample_collection.hnsw_index.nodes

def test_collection_remove_vector(sample_collection):
    """Test removing vectors from a collection."""
    # Verify vec1 exists
    assert "vec1" in sample_collection.store.vectors
    
    # Remove it
    sample_collection.remove_vector("vec1")
    
    # Verify it's gone from the store
    assert "vec1" not in sample_collection.store.vectors
    
    # Verify it's gone from the HNSW index if enabled
    if sample_collection.use_hnsw and sample_collection.hnsw_index:
        assert "vec1" not in sample_collection.hnsw_index.nodes

def test_collection_search_similar(sample_collection):
    """Test searching for similar vectors in a collection."""
    # Search for vectors similar to [1.0, 0.1, 0.0]
    results = sample_collection.search_similar([1.0, 0.1, 0.0], k=2)
    
    assert len(results) == 2
    # Check that we get valid results with similarity scores
    result_ids = [id for id, _ in results]
    assert len(result_ids) == 2
    # Check that similarity scores are positive
    for _, score in results:
        assert score >= 0.0
    
    # Test with k parameter
    limited_results = sample_collection.search_similar([1.0, 0.1, 0.0], k=1)
    assert len(limited_results) == 1

def test_collection_get_vector_count(sample_collection):
    """Test getting the vector count from a collection."""
    assert sample_collection.get_vector_count() == 4
    
    # Add another vector
    sample_collection.add_vector("vec5", [0.5, 0.5, 0.5], {"tag": "gray"})
    assert sample_collection.get_vector_count() == 5
    
    # Remove a vector
    sample_collection.remove_vector("vec1")
    assert sample_collection.get_vector_count() == 4

def test_collection_manager_initialization(temp_dir):
    """Test that CollectionManager initializes correctly."""
    manager = CollectionManager(temp_dir)
    
    assert manager.base_storage_dir == temp_dir
    assert manager.collections_dir == os.path.join(temp_dir, "collections")
    assert os.path.exists(manager.collections_dir)
    assert manager.use_hnsw is True
    assert manager.collections == {}
    assert manager.collections_metadata == {}

def test_collection_manager_create_collection(collection_manager):
    """Test creating collections with the CollectionManager."""
    # Create a collection
    collection = collection_manager.create_collection("test1")
    
    # Verify it was created
    assert "test1" in collection_manager.collections
    assert "test1" in collection_manager.collections_metadata
    assert collection.name == "test1"
    assert collection.use_hnsw is True
    
    # Create another collection with custom HNSW settings
    collection2 = collection_manager.create_collection(
        "test2", 
        use_hnsw=False, 
        hnsw_params={"M": 20, "ef_construction": 300}
    )
    
    assert "test2" in collection_manager.collections
    assert collection2.use_hnsw is False
    assert collection2.hnsw_params["M"] == 20
    
    # Test that creating an existing collection raises ValueError
    with pytest.raises(ValueError):
        collection_manager.create_collection("test1")

def test_collection_manager_get_collection(collection_manager, sample_collection):
    """Test getting collections from the CollectionManager."""
    # Check the existing collection is accessible
    retrieved = collection_manager.get_collection("test_collection")
    assert retrieved is sample_collection
    
    # Test getting a non-existent collection raises KeyError
    with pytest.raises(KeyError):
        collection_manager.get_collection("non_existent")

def test_collection_manager_list_collections(collection_manager):
    """Test listing collections from the CollectionManager."""
    # Create a new collection for testing
    collection_manager.create_collection("test_list_collection")
    
    # Should have the new collection
    collections = collection_manager.list_collections()
    assert len(collections) >= 1
    assert "test_list_collection" in collections

def test_collection_manager_delete_collection(collection_manager):
    """Test deleting collections with the CollectionManager."""
    # Create a collection to delete
    collection_manager.create_collection("to_delete")
    assert "to_delete" in collection_manager.list_collections()
    
    # Delete it
    collection_manager.delete_collection("to_delete")
    
    # Verify it's gone
    assert "to_delete" not in collection_manager.list_collections()
    assert "to_delete" not in collection_manager.collections
    assert "to_delete" not in collection_manager.collections_metadata
    
    # Test that deleting a non-existent collection raises KeyError
    with pytest.raises(KeyError):
        collection_manager.delete_collection("non_existent")

def test_collection_manager_get_collection_stats(collection_manager, sample_collection):
    """Test getting collection statistics from the CollectionManager."""
    stats = collection_manager.get_collection_stats("test_collection")
    
    assert stats["name"] == "test_collection"
    assert stats["vector_count"] == 4
    assert stats["dimension"] == 3
    assert stats["using_hnsw"] is True
    assert "vectors_size_bytes" in stats
    assert "metadata_size_bytes" in stats
    assert "total_size_bytes" in stats 