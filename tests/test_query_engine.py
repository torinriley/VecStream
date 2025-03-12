import pytest
import numpy as np
from vector_db.vector_store import VectorStore
from vector_db.index_manager import IndexManager
from vector_db.query_engine import QueryEngine

def test_vector_similarity():
    store = VectorStore()
    index_manager = IndexManager(store)
    query_engine = QueryEngine(store, index_manager)
    
    # Test cosine similarity
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert query_engine.vector_similarity(vec1, vec2, "cosine") == 1.0
    
    # Orthogonal vectors
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    assert query_engine.vector_similarity(vec1, vec2, "cosine") == 0.0
    
    # Test euclidean similarity
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    assert query_engine.vector_similarity(vec1, vec2, "euclidean") == 1.0

def test_search():
    store = VectorStore()
    store.add("vec1", [1.0, 0.0, 0.0])
    store.add("vec2", [0.0, 1.0, 0.0])
    store.add("vec3", [0.0, 0.0, 1.0])
    store.add("vec4", [0.9, 0.1, 0.0])
    
    index_manager = IndexManager(store)
    query_engine = QueryEngine(store, index_manager)
    
    # Search closest to [1,0,0]
    results = query_engine.search([1.0, 0.0, 0.0], 2)
    
    assert len(results) == 2
    assert results[0][0] == "vec1"  # Exact match should be first
    assert results[1][0] == "vec4"  # Second closest
    
    # Verify similarity scores are in descending order
    assert results[0][1] > results[1][1]
