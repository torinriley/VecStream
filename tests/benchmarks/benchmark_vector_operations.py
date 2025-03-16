"""
Benchmark tests for VecStream vector operations.
"""
import time
import numpy as np
from vecstream import VectorStore

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
    for num_vectors in [1000, 10000, 100000]:
        result = benchmark_add_vectors(num_vectors=num_vectors)
        results["benchmarks"].append(result)
    
    # Run search benchmark with different sizes
    for num_vectors in [1000, 10000, 100000]:
        result = benchmark_search(num_vectors=num_vectors)
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