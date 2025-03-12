import pytest
from vector_db.vector_store import VectorStore
from vector_db.index_manager import IndexManager, IndexType

def test_index_creation():
    store = VectorStore()
    store.add("vec1", [1.0, 2.0, 3.0])
    store.add("vec2", [4.0, 5.0, 6.0])
    
    index_manager = IndexManager(store)
    
    assert index_manager.is_indexed is False
    
    result = index_manager.build_index()
    
    assert result is True
    assert index_manager.is_indexed is True
    assert index_manager.index_type == IndexType.FLAT

def test_index_types():
    store = VectorStore()
    index_manager = IndexManager(store)
    
    # Test flat index
    result = index_manager.build_index(IndexType.FLAT)
    assert result is True
    assert index_manager.index_type == IndexType.FLAT
    
    # The other index types would be more complex in implementation,
    # but we can test the basic interface
    assert IndexType.IVF.value == "ivf"
    assert IndexType.HNSW.value == "hnsw"

def test_update_index():
    store = VectorStore()
    store.add("vec1", [1.0, 2.0, 3.0])
    
    index_manager = IndexManager(store)
    index_manager.build_index()
    
    # Update the index with a new vector
    result = index_manager.update_index("vec2", [4.0, 5.0, 6.0])
    assert result is True
