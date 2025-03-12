import pytest
import numpy as np
from vector_db.vector_store import VectorStore

def test_add_vector():
    store = VectorStore()
    result = store.add("test1", [1.0, 2.0, 3.0])
    
    assert result is True
    assert "test1" in store.vectors
    np.testing.assert_array_almost_equal(store.vectors["test1"], np.array([1.0, 2.0, 3.0]))

def test_add_vector_with_metadata():
    store = VectorStore()
    metadata = {"category": "test", "importance": 5}
    store.add("test2", [4.0, 5.0, 6.0], metadata)
    
    assert "test2" in store.vectors
    assert "test2" in store.metadata
    assert store.metadata["test2"] == metadata

def test_get_vector():
    store = VectorStore()
    store.add("test3", [7.0, 8.0, 9.0])
    
    result = store.get("test3")
    np.testing.assert_array_almost_equal(result, np.array([7.0, 8.0, 9.0]))
    
    assert store.get("nonexistent") is None

def test_delete_vector():
    store = VectorStore()
    store.add("test4", [1.0, 1.0, 1.0])
    
    assert store.delete("test4") is True
    assert "test4" not in store.vectors
    
    assert store.delete("nonexistent") is False

def test_list_ids():
    store = VectorStore()
    store.add("id1", [1.0, 0.0, 0.0])
    store.add("id2", [0.0, 1.0, 0.0])
    
    ids = store.list_ids()
    assert len(ids) == 2
    assert "id1" in ids
    assert "id2" in ids

def test_count():
    store = VectorStore()
    assert store.count() == 0
    
    store.add("id1", [1.0, 0.0, 0.0])
    store.add("id2", [0.0, 1.0, 0.0])
    
    assert store.count() == 2
