"""
Unit tests for the HNSW index implementation.
"""
import pytest
import numpy as np
from vecstream import HNSWIndex

@pytest.fixture
def sample_hnsw_index():
    """Fixture that provides a small HNSW index with sample vectors."""
    index = HNSWIndex(dim=3, M=10, ef_construction=100)
    
    # Add some sample vectors
    vectors = [
        ("vec1", np.array([1.0, 0.0, 0.0])),
        ("vec2", np.array([0.0, 1.0, 0.0])),
        ("vec3", np.array([0.0, 0.0, 1.0])),
        ("vec4", np.array([0.7, 0.7, 0.0])),
        ("vec5", np.array([0.3, 0.3, 0.3])),
    ]
    
    for id, vector in vectors:
        index.add_item(id, vector)
    
    return index

def test_hnsw_index_initialization():
    """Test that HNSW index initializes correctly."""
    index = HNSWIndex(dim=5, M=16, ef_construction=200, ml=4)
    
    assert index.dim == 5
    assert index.M == 16
    assert index.M_max0 == 32  # Should be 2*M
    assert index.ef_construction == 200
    assert index.ml == 4
    assert index.nodes == {}
    assert index.node_levels == {}
    assert len(index.graphs) == 5  # ml + 1 levels

def test_add_item(sample_hnsw_index):
    """Test adding items to the HNSW index."""
    # Test that existing vectors are in the index
    assert "vec1" in sample_hnsw_index.nodes
    assert "vec2" in sample_hnsw_index.nodes
    assert "vec3" in sample_hnsw_index.nodes
    
    # Add a new vector
    new_vec = np.array([0.5, 0.5, 0.5])
    sample_hnsw_index.add_item("new_vec", new_vec)
    
    # Verify it was added
    assert "new_vec" in sample_hnsw_index.nodes
    np.testing.assert_array_equal(sample_hnsw_index.nodes["new_vec"], new_vec)
    
    # Check that nodes dictionary has the item
    assert "new_vec" in sample_hnsw_index.nodes

def test_remove_item(sample_hnsw_index):
    """Test removing items from the HNSW index."""
    # Verify vec1 exists before removal
    assert "vec1" in sample_hnsw_index.nodes
    
    # Remove vec1
    sample_hnsw_index.remove_item("vec1")
    
    # Verify it was removed from nodes, levels, and graphs
    assert "vec1" not in sample_hnsw_index.nodes
    assert "vec1" not in sample_hnsw_index.node_levels
    
    # Check it's removed from all graph levels
    for level in sample_hnsw_index.graphs:
        assert "vec1" not in sample_hnsw_index.graphs[level]
        
        # Also verify no other nodes have connections to vec1
        for node_id, neighbors in sample_hnsw_index.graphs[level].items():
            assert "vec1" not in neighbors
    
    # Test removing non-existent item raises KeyError
    with pytest.raises(KeyError):
        sample_hnsw_index.remove_item("non_existent_vec")

def test_search(sample_hnsw_index):
    """Test searching for nearest neighbors in the HNSW index."""
    # Search for the nearest neighbor to [1.0, 0.1, 0.0]
    query = np.array([1.0, 0.1, 0.0])
    results = sample_hnsw_index.search(query, k=1)
    
    # Just check that we get a result with valid ID
    assert len(results) == 1
    assert isinstance(results[0][0], str)  # ID should be a string
    
    # Search for nearest neighbors to [0.5, 0.5, 0.0]
    query2 = np.array([0.5, 0.5, 0.0])
    results2 = sample_hnsw_index.search(query2, k=5)
    
    # Check that we get multiple results
    assert len(results2) > 0
    
    # Test searching with custom ef_search parameter
    results3 = sample_hnsw_index.search(query2, k=3, ef_search=50)
    assert len(results3) <= 3  # May return fewer if not enough vectors

def test_search_empty_index():
    """Test searching an empty index returns empty results."""
    empty_index = HNSWIndex(dim=3)
    results = empty_index.search(np.array([1.0, 0.0, 0.0]), k=5)
    assert len(results) == 0

def test_distance_calculation(sample_hnsw_index):
    """Test the distance calculation used by HNSW."""
    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([0.0, 1.0, 0.0])
    
    # Cosine distance between orthogonal vectors should be 1.0
    distance = sample_hnsw_index._distance(vec1, vec2)
    assert distance == pytest.approx(1.0)
    
    # Distance to self should be 0.0
    self_distance = sample_hnsw_index._distance(vec1, vec1)
    assert self_distance == pytest.approx(0.0)
    
    # Test with non-normalized vectors
    vec3 = np.array([2.0, 0.0, 0.0])
    vec4 = np.array([0.0, 2.0, 0.0])
    distance2 = sample_hnsw_index._distance(vec3, vec4)
    assert distance2 == pytest.approx(1.0)
    
    # Test with zero vector
    zero_vec = np.array([0.0, 0.0, 0.0])
    zero_distance = sample_hnsw_index._distance(vec1, zero_vec)
    assert zero_distance == pytest.approx(1.0)  # Should clamp to reasonable distance 