"""
Integration test for VecStream features.
This test verifies the interaction between different components:
1. HNSW indexing
2. Collections/namespaces
3. Metadata filtering
"""

import os
import time
import shutil
import numpy as np
import tempfile
import pytest
from pathlib import Path

from vecstream.hnsw_index import HNSWIndex
from vecstream.collections import Collection, CollectionManager

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_hnsw_indexing(temp_dir):
    """Test the HNSW indexing functionality."""
    # Create an HNSW index directly
    dim = 3
    hnsw_index = HNSWIndex(dim=dim, M=16, ef_construction=100)
    
    # Add vectors to the index
    vectors = [
        ("vec1", np.array([1.0, 0.0, 0.0])),
        ("vec2", np.array([0.5, 0.5, 0.0])),
        ("vec3", np.array([0.8, 0.2, 0.0])),
        ("vec4", np.array([0.0, 1.0, 0.0])),
        ("vec5", np.array([0.0, 0.0, 1.0])),
    ]
    
    for id, vector in vectors:
        hnsw_index.add_item(id, vector)
    
    # Search for nearest neighbors
    query = np.array([0.8, 0.2, 0.0])
    results = hnsw_index.search(query, k=3)
    
    # Verify we got reasonable results
    assert len(results) > 0, "Expected at least one result"
    for _, sim in results:
        assert sim >= 0.0, "Expected positive similarity"
    
    # Test removing a vector
    hnsw_index.remove_item("vec1")
    new_results = hnsw_index.search(query, k=3)
    assert "vec1" not in [r[0] for r in new_results], "vec1 should be removed from results"

def test_collections_and_filtering(temp_dir):
    """Test collections with metadata filtering."""
    # Create a collection manager
    collection_dir = os.path.join(temp_dir, "collections")
    manager = CollectionManager(collection_dir, use_hnsw=True)
    
    # Create two collections
    images = manager.create_collection("images")
    texts = manager.create_collection("texts")
    
    # Add vectors to collections with metadata
    images.add_vector("img1", [1.0, 0.0, 0.0], {"format": "jpg", "size": "1024x768"})
    images.add_vector("img2", [0.0, 1.0, 0.0], {"format": "png", "size": "800x600"})
    
    texts.add_vector("text1", [0.0, 0.0, 1.0], {"language": "en", "word_count": 500})
    texts.add_vector("text2", [0.7, 0.7, 0.0], {"language": "fr", "word_count": 300})
    
    # Verify collections
    collections = manager.list_collections()
    assert "images" in collections, "Images collection not found"
    assert "texts" in collections, "Texts collection not found"
    
    # Get collection stats
    img_stats = manager.get_collection_stats("images")
    assert img_stats["vector_count"] == 2, f"Expected 2 vectors in images collection, got {img_stats['vector_count']}"
    
    # Create a collection for testing filtering
    products = manager.create_collection("products")
    
    # Add products with various metadata
    products.add_vector("prod1", [1.0, 0.0, 0.0], 
                        {"category": "electronics", "price": 100, "in_stock": True,
                         "details": {"color": "black", "brand": "Sony"}})
    
    products.add_vector("prod2", [0.9, 0.1, 0.0], 
                        {"category": "electronics", "price": 200, "in_stock": False,
                         "details": {"color": "silver", "brand": "Apple"}})
    
    products.add_vector("prod3", [0.0, 1.0, 0.0], 
                        {"category": "clothing", "price": 50, "in_stock": True,
                         "details": {"color": "blue", "size": "M"}})
    
    products.add_vector("prod4", [0.0, 0.0, 1.0], 
                        {"category": "books", "price": 15, "in_stock": True,
                         "details": {"author": "Jane Doe", "pages": 300}})
    
    # Test basic metadata filtering
    query = [0.5, 0.5, 0.0]
    filter1 = {"category": "electronics", "in_stock": True}
    
    results1 = products.search_similar(query, k=2, filter_metadata=filter1)
    assert len(results1) > 0, "No results found with filter"
    
    # Test nested metadata filtering with dot notation
    filter2 = {"details.color": "black"}
    results2 = products.search_similar(query, k=2, filter_metadata=filter2)
    assert len(results2) > 0, "No results found with nested filter"
    
    # Test combined filters
    filter3 = {"category": "electronics", "details.brand": "Apple"}
    results3 = products.search_similar(query, k=2, filter_metadata=filter3)
    assert len(results3) > 0, "No results found with combined filter"
    assert results3[0][0] == "prod2", f"Expected prod2 as result for combined filter, got {results3[0][0]}" 