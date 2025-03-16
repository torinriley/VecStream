"""
Unit tests for the VectorStore class.
"""
import pytest
import numpy as np
from vecstream import VectorStore

def test_vector_store_initialization():
    """Test that VectorStore initializes correctly."""
    store = VectorStore()
    assert store.vectors == {}
    assert store.dimension is None

def test_add_vector(vector_store, sample_vectors):
    """Test adding vectors to the store."""
    vec_id, vector = sample_vectors[0]
    vector_store.add_vector(vec_id, vector)
    assert vec_id in vector_store.vectors
    np.testing.assert_array_equal(vector_store.vectors[vec_id], vector)
    assert vector_store.dimension == len(vector)

def test_add_multiple_vectors(vector_store, sample_vectors):
    """Test adding multiple vectors to the store."""
    for vec_id, vector in sample_vectors:
        vector_store.add_vector(vec_id, vector)
    
    assert len(vector_store.vectors) == len(sample_vectors)
    for vec_id, vector in sample_vectors:
        assert vec_id in vector_store.vectors
        np.testing.assert_array_equal(vector_store.vectors[vec_id], vector)

def test_search_similar(vector_store, sample_vectors):
    """Test searching for similar vectors."""
    # Add all sample vectors
    for vec_id, vector in sample_vectors:
        vector_store.add_vector(vec_id, vector)
    
    # Search for a vector similar to vec1
    query = [1.0, 0.1, 0.1]
    results = vector_store.search_similar(query, k=2)
    
    assert len(results) == 2
    assert results[0][0] == "vec1"  # Most similar should be vec1 