#!/usr/bin/env python
"""
Advanced features example for VecStream, demonstrating:
1. HNSW indexing for fast similarity search
2. Collections/namespaces for organizing vectors
3. Metadata filtering for fine-grained search
"""

import time
import numpy as np
import os
from pathlib import Path
from rich.console import Console
from rich.table import Table

from vecstream.collections import CollectionManager
from vecstream.hnsw_index import HNSWIndex

# Create a console for pretty output
console = Console()

# Create a temporary directory for this example
temp_dir = os.path.join(Path.home(), ".vecstream_example")
os.makedirs(temp_dir, exist_ok=True)

console.print("\n[bold cyan]VecStream Advanced Features Example[/bold cyan]")
console.print("[dim]This example demonstrates HNSW indexing, collections, and metadata filtering[/dim]\n")

# Initialize the collection manager
manager = CollectionManager(temp_dir, use_hnsw=True)

# Create collections for different vector types
science_collection = manager.create_collection("science")
tech_collection = manager.create_collection("technology")
art_collection = manager.create_collection("art")

console.print("[green]Created three collections: science, technology, art[/green]")

# Generate some random document vectors and metadata
def create_random_vectors(n=1000, dim=128):
    """Create random normalized vectors."""
    vectors = np.random.randn(n, dim)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors

# Generate vectors
console.print("[yellow]Generating random vectors...[/yellow]")
n_vectors = 10000
dim = 128
all_vectors = create_random_vectors(n_vectors, dim)

# Split vectors between collections
science_vectors = all_vectors[:3000]
tech_vectors = all_vectors[3000:7000]
art_vectors = all_vectors[7000:]

# Add vectors to collections with appropriate metadata
console.print("[yellow]Adding vectors to collections with metadata...[/yellow]")

# Science collection
for i in range(len(science_vectors)):
    category = np.random.choice(["physics", "biology", "chemistry"])
    year = np.random.randint(2000, 2024)
    metadata = {
        "category": category,
        "year": year,
        "is_peer_reviewed": bool(np.random.choice([True, False], p=[0.8, 0.2])),
        "details": {
            "citations": np.random.randint(0, 1000),
            "journal": np.random.choice(["Nature", "Science", "PNAS", "PLoS ONE"])
        }
    }
    science_collection.add_vector(f"sci_{i}", science_vectors[i].tolist(), metadata)

# Technology collection
for i in range(len(tech_vectors)):
    category = np.random.choice(["ai", "web", "mobile", "cloud"])
    year = np.random.randint(2010, 2024)
    metadata = {
        "category": category,
        "year": year,
        "is_open_source": bool(np.random.choice([True, False])),
        "details": {
            "stars": np.random.randint(0, 50000),
            "language": np.random.choice(["Python", "JavaScript", "Go", "Rust"])
        }
    }
    tech_collection.add_vector(f"tech_{i}", tech_vectors[i].tolist(), metadata)

# Art collection
for i in range(len(art_vectors)):
    category = np.random.choice(["painting", "sculpture", "digital", "photography"])
    year = np.random.randint(1800, 2024)
    metadata = {
        "category": category,
        "year": year,
        "is_exhibited": bool(np.random.choice([True, False])),
        "details": {
            "artist": np.random.choice(["Anonymous", "Smith", "Johnson", "Garcia", "Wong"]),
            "medium": np.random.choice(["Oil", "Acrylic", "Digital", "Mixed"])
        }
    }
    art_collection.add_vector(f"art_{i}", art_vectors[i].tolist(), metadata)

console.print("[green]Added vectors to collections with rich metadata![/green]")

# Show collection stats
console.print("\n[bold cyan]Collection Statistics:[/bold cyan]")
table = Table()
table.add_column("Collection", style="cyan")
table.add_column("Vectors", style="green")
table.add_column("Using HNSW", style="yellow")

for collection_name in ["science", "technology", "art"]:
    stats = manager.get_collection_stats(collection_name)
    table.add_row(
        collection_name,
        str(stats["vector_count"]),
        "Yes" if stats["using_hnsw"] else "No"
    )

console.print(table)

# Benchmark search performance
console.print("\n[bold cyan]Search Performance Benchmark:[/bold cyan]")

# Create a query vector (normalized random vector)
query = np.random.randn(dim)
query = query / np.linalg.norm(query)

# Get the technology collection
tech_coll = manager.get_collection("technology")

# Benchmark HNSW search
console.print("\n[yellow]Benchmarking HNSW search...[/yellow]")
start_time = time.time()
hnsw_results = tech_coll.search_similar(query.tolist(), k=10)
hnsw_time = time.time() - start_time
console.print(f"[green]HNSW search time: {hnsw_time:.6f} seconds[/green]")

# Benchmark standard search (by temporarily disabling HNSW)
console.print("\n[yellow]Benchmarking standard search...[/yellow]")
# Store the HNSW index and temporarily disable it
saved_index = tech_coll.hnsw_index
tech_coll.hnsw_index = None
tech_coll.use_hnsw = False

start_time = time.time()
standard_results = tech_coll.search_similar(query.tolist(), k=10)
standard_time = time.time() - start_time
console.print(f"[green]Standard search time: {standard_time:.6f} seconds[/green]")

# Restore HNSW index
tech_coll.hnsw_index = saved_index
tech_coll.use_hnsw = True

# Speed comparison
speedup = standard_time / hnsw_time if hnsw_time > 0 else float('inf')
console.print(f"[bold green]HNSW speedup: {speedup:.2f}x faster![/bold green]")

# Demonstrate metadata filtering
console.print("\n[bold cyan]Metadata Filtering Examples:[/bold cyan]")

# Example 1: Filter science papers by category and year
filter1 = {"category": "physics", "year": 2022}
console.print(f"\n[yellow]Searching science collection with filter: {filter1}[/yellow]")
start_time = time.time()
results1 = science_collection.search_similar(
    query.tolist(), 
    k=5, 
    filter_metadata=filter1
)
filter_time = time.time() - start_time

# Print results
table = Table(title="Physics papers from 2022")
table.add_column("ID", style="cyan")
table.add_column("Similarity", style="green")
table.add_column("Metadata", style="white")

for id, score in results1:
    _, metadata = science_collection.get_vector_with_metadata(id)
    table.add_row(id, f"{score:.4f}", str(metadata))

console.print(table)
console.print(f"[green]Filter search time: {filter_time:.6f} seconds[/green]")

# Example 2: Filter technology repositories by nested property
filter2 = {"details.language": "Python", "is_open_source": True}
console.print(f"\n[yellow]Searching technology collection with nested filter: {filter2}[/yellow]")
results2 = tech_collection.search_similar(
    query.tolist(), 
    k=5, 
    filter_metadata=filter2
)

# Print results
table = Table(title="Open source Python projects")
table.add_column("ID", style="cyan")
table.add_column("Similarity", style="green")
table.add_column("Stars", style="yellow")
table.add_column("Year", style="blue")

for id, score in results2:
    _, metadata = tech_collection.get_vector_with_metadata(id)
    table.add_row(
        id, 
        f"{score:.4f}", 
        str(metadata["details"]["stars"]),
        str(metadata["year"])
    )

console.print(table)

# Clean up
console.print("\n[yellow]Do you want to keep the example data? (y/n)[/yellow]")
response = input().lower()
if response != 'y':
    import shutil
    shutil.rmtree(temp_dir)
    console.print("[green]Example data cleaned up.[/green]")
else:
    console.print(f"[green]Example data kept at: {temp_dir}[/green]")

console.print("\n[bold cyan]Example completed![/bold cyan]") 