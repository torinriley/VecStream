"""
Benchmark tests for VecStream vector operations.
"""
import time
import os
import numpy as np
import tempfile
import shutil
from vecstream import VectorStore
from vecstream.hnsw_index import HNSWIndex
from vecstream.collections import Collection, CollectionManager
from vecstream.persistent_store import PersistentVectorStore

def generate_random_vectors(num_vectors, dimension):
    """Generate random vectors for benchmarking."""
    vectors = []
    for i in range(num_vectors):
        vec = np.random.randn(dimension)
        vec = vec / np.linalg.norm(vec)  # Normalize
        vectors.append((f"vec_{i}", vec.tolist()))
    return vectors

def benchmark_add_vectors(num_vectors=10000, dimension=384):
    """Benchmark vector addition performance."""
    store = VectorStore()
    vectors = generate_random_vectors(num_vectors, dimension)
    
    start_time = time.time()
    for vec_id, vector in vectors:
        store.add_vector(vec_id, vector)
    end_time = time.time()
    
    add_time = end_time - start_time
    vectors_per_second = num_vectors / add_time
    
    return {
        "operation": "add_vectors",
        "num_vectors": num_vectors,
        "dimension": dimension,
        "total_time": add_time,
        "vectors_per_second": vectors_per_second
    }

def benchmark_search(num_vectors=10000, dimension=384, k=10, num_queries=100):
    """Benchmark similarity search performance."""
    # Setup
    store = VectorStore()
    vectors = generate_random_vectors(num_vectors, dimension)
    for vec_id, vector in vectors:
        store.add_vector(vec_id, vector)
    
    # Generate random queries
    queries = [vec for _, vec in generate_random_vectors(num_queries, dimension)]
    
    # Benchmark search
    start_time = time.time()
    for query in queries:
        store.search_similar(query, k=k)
    end_time = time.time()
    
    search_time = end_time - start_time
    queries_per_second = num_queries / search_time
    
    return {
        "operation": "search",
        "num_vectors": num_vectors,
        "dimension": dimension,
        "k": k,
        "num_queries": num_queries,
        "total_time": search_time,
        "queries_per_second": queries_per_second
    }

def benchmark_hnsw_index(num_vectors=10000, dimension=128, k=10, num_queries=100, M=16, ef_construction=200):
    """Benchmark HNSW index performance."""
    # Setup
    hnsw_index = HNSWIndex(dim=dimension, M=M, ef_construction=ef_construction)
    vectors = generate_random_vectors(num_vectors, dimension)
    
    # Benchmark add items
    start_time = time.time()
    for vec_id, vector in vectors:
        hnsw_index.add_item(vec_id, np.array(vector))
    add_time = time.time() - start_time
    
    # Generate random queries
    queries = [np.array(vec) for _, vec in generate_random_vectors(num_queries, dimension)]
    
    # Benchmark search
    start_time = time.time()
    for query in queries:
        hnsw_index.search(query, k=k)
    search_time = time.time() - start_time
    
    return {
        "operation": "hnsw_index",
        "num_vectors": num_vectors,
        "dimension": dimension,
        "M": M,
        "ef_construction": ef_construction,
        "k": k,
        "num_queries": num_queries,
        "add_time": add_time,
        "search_time": search_time,
        "vectors_per_second_add": num_vectors / add_time,
        "queries_per_second": num_queries / search_time
    }

def benchmark_collections(num_vectors=5000, dimension=128, k=10, num_queries=50):
    """Benchmark collections performance."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Setup
        collection_dir = os.path.join(temp_dir, "collections")
        manager = CollectionManager(collection_dir, use_hnsw=True)
        collection = manager.create_collection("benchmark_collection")
        
        vectors = generate_random_vectors(num_vectors, dimension)
        
        # Create metadata for vectors
        metadata_list = []
        for i in range(num_vectors):
            metadata = {
                "category": f"category_{i % 5}",
                "price": i % 100 + 10,
                "in_stock": i % 2 == 0,
                "details": {
                    "color": f"color_{i % 10}",
                    "brand": f"brand_{i % 8}"
                }
            }
            metadata_list.append(metadata)
        
        # Benchmark add vectors with metadata
        start_time = time.time()
        for i, (vec_id, vector) in enumerate(vectors):
            collection.add_vector(vec_id, vector, metadata_list[i])
        add_time = time.time() - start_time
        
        # Generate random queries
        queries = [vec for _, vec in generate_random_vectors(num_queries, dimension)]
        
        # Benchmark simple search
        start_time = time.time()
        for query in queries:
            collection.search_similar(query, k=k)
        search_time = time.time() - start_time
        
        # Benchmark search with filter
        start_time = time.time()
        for query in queries:
            collection.search_similar(query, k=k, filter_metadata={"category": "category_1"})
        filter_search_time = time.time() - start_time
        
        # Benchmark search with nested filter
        start_time = time.time()
        for query in queries:
            collection.search_similar(query, k=k, filter_metadata={"details.brand": "brand_2"})
        nested_filter_time = time.time() - start_time
        
        return {
            "operation": "collections",
            "num_vectors": num_vectors,
            "dimension": dimension,
            "k": k,
            "num_queries": num_queries,
            "add_time": add_time,
            "search_time": search_time,
            "filter_search_time": filter_search_time,
            "nested_filter_time": nested_filter_time,
            "vectors_per_second_add": num_vectors / add_time,
            "queries_per_second": num_queries / search_time,
            "filter_queries_per_second": num_queries / filter_search_time,
            "nested_filter_queries_per_second": num_queries / nested_filter_time
        }
    finally:
        shutil.rmtree(temp_dir)

def run_benchmarks():
    """Run all benchmarks and save results."""
    import json
    import os
    from datetime import datetime
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": []
    }
    
    # Run add vectors benchmark with different sizes
    for num_vectors in [1000, 10000]:
        result = benchmark_add_vectors(num_vectors=num_vectors)
        results["benchmarks"].append(result)
    
    # Run search benchmark with different sizes
    for num_vectors in [1000, 10000]:
        result = benchmark_search(num_vectors=num_vectors)
        results["benchmarks"].append(result)
    
    # Run HNSW index benchmark
    for num_vectors in [1000, 10000]:
        result = benchmark_hnsw_index(num_vectors=num_vectors, dimension=128)
        results["benchmarks"].append(result)
    
    # Run collections benchmark
    result = benchmark_collections(num_vectors=5000, dimension=128)
    results["benchmarks"].append(result)
    
    # Save results
    os.makedirs("tests/benchmarks/results", exist_ok=True)
    filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join("tests/benchmarks/results", filename)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    results = run_benchmarks()
    print("Benchmark results:")
    for benchmark in results["benchmarks"]:
        print(f"\n{benchmark['operation']}:")
        for key, value in benchmark.items():
            if key != "operation":
                print(f"  {key}: {value}") 