"""
Example script demonstrating binary vector storage with VecStream.
"""

from vecstream.binary_store import BinaryVectorStore
import numpy as np
import os
from rich.console import Console
from rich.table import Table

console = Console()

def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

def main():
    # Create a binary store in the 'vector_db' directory
    store_dir = "vector_db"
    store = BinaryVectorStore(store_dir)
    
    # Sample documents with metadata
    documents = [
        {
            "text": "Machine learning is a subset of artificial intelligence",
            "tags": ["AI", "ML"],
            "category": "technology"
        },
        {
            "text": "Python is a popular programming language",
            "tags": ["programming", "Python"],
            "category": "software"
        },
        {
            "text": "Neural networks are inspired by biological brains",
            "tags": ["AI", "neuroscience"],
            "category": "science"
        },
        {
            "text": "Data science involves statistical analysis",
            "tags": ["data", "statistics"],
            "category": "analytics"
        },
        {
            "text": "Deep learning is revolutionizing AI",
            "tags": ["AI", "deep learning"],
            "category": "technology"
        }
    ]
    
    console.print("\n🔄 [bold cyan]Adding documents to binary store...[/bold cyan]")
    # Add documents with vectors and metadata
    for i, doc in enumerate(documents):
        # Simulate document embedding (normally you'd use a proper embedding model)
        vector = np.random.random(384).astype(np.float32).tolist()
        doc_id = f"doc_{i}"
        
        # Add vector with metadata
        store.add_vector(
            id=doc_id,
            vector=vector,
            metadata={
                "text": doc["text"],
                "tags": doc["tags"],
                "category": doc["category"]
            }
        )
    
    # Display store contents
    console.print("\n📚 [bold cyan]Current documents in store:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim")
    table.add_column("Text")
    table.add_column("Tags", style="cyan")
    table.add_column("Category", style="green")
    
    for i in range(len(documents)):
        doc_id = f"doc_{i}"
        vector, metadata = store.get_vector_with_metadata(doc_id)
        table.add_row(
            doc_id,
            metadata["text"],
            ", ".join(metadata["tags"]),
            metadata["category"]
        )
    
    console.print(table)
    
    # Perform similarity search
    console.print("\n🔍 [bold cyan]Performing similarity search...[/bold cyan]")
    query_vector = np.random.random(384).astype(np.float32).tolist()
    results = store.search_similar(query_vector, k=3)
    
    # Display search results
    console.print("\n[bold cyan]Top 3 similar documents:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Score", style="cyan")
    table.add_column("Text")
    table.add_column("Category", style="green")
    
    for doc_id, score in results:
        _, metadata = store.get_vector_with_metadata(doc_id)
        table.add_row(
            f"{score:.4f}",
            metadata["text"],
            metadata["category"]
        )
    
    console.print(table)
    
    # Display storage information
    vectors_size, metadata_size = store.get_store_size()
    total_size = vectors_size + metadata_size
    
    console.print("\n💾 [bold cyan]Storage Information:[/bold cyan]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Size", style="cyan")
    table.add_column("Percentage", style="green")
    
    table.add_row(
        "Vectors File",
        format_size(vectors_size),
        f"{(vectors_size/total_size)*100:.1f}%"
    )
    table.add_row(
        "Metadata File",
        format_size(metadata_size),
        f"{(metadata_size/total_size)*100:.1f}%"
    )
    table.add_row(
        "Total Size",
        format_size(total_size),
        "100%"
    )
    
    console.print(table)
    
    # Show storage location
    console.print(f"\n📁 [bold cyan]Storage Location:[/bold cyan] {os.path.abspath(store_dir)}")

if __name__ == "__main__":
    main() 