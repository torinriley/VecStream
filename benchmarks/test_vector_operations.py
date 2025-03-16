"""Benchmark tests for vector operations."""

import numpy as np
from vecstream import VectorStore

def generate_random_vectors(num_vectors, dimension):
    """Generate random vectors for testing."""
    return {
        f"vec_{i}": np.random.random(dimension).astype(np.float32)
        for i in range(num_vectors)
    }

def test_add_vector_benchmark(benchmark):
    """Benchmark vector addition performance."""
    store = VectorStore()
    vector = np.random.random(384).astype(np.float32)
    
    def add_vector():
        store.add_vector("test_vec", vector)
    
    benchmark(add_vector)

def test_search_similar_benchmark(benchmark):
    """Benchmark similarity search performance."""
    store = VectorStore()
    dimension = 384
    num_vectors = 10000
    
    # Add test vectors
    vectors = generate_random_vectors(num_vectors, dimension)
    for vec_id, vector in vectors.items():
        store.add_vector(vec_id, vector)
    
    # Create query vector
    query = np.random.random(dimension).astype(np.float32)
    
    def search_similar():
        store.search_similar(query, k=10)
    
    benchmark(search_similar)

def test_bulk_add_benchmark(benchmark):
    """Benchmark bulk vector addition performance."""
    store = VectorStore()
    dimension = 384
    num_vectors = 1000
    vectors = generate_random_vectors(num_vectors, dimension)
    
    def bulk_add():
        for vec_id, vector in vectors.items():
            store.add_vector(vec_id, vector)
    
    benchmark(bulk_add)

def test_vector_normalization_benchmark(benchmark):
    """Benchmark vector normalization performance."""
    dimension = 384
    num_vectors = 1000
    vectors = list(generate_random_vectors(num_vectors, dimension).values())
    
    def normalize_vectors():
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / norms
        return normalized
    
    benchmark(normalize_vectors) 