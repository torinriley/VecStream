"""
VecStream CLI - A lightweight vector database with similarity search.
"""

import click
from rich import print
from rich.table import Table
from rich.progress import track
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from pathlib import Path

from .vector_store import VectorStore
from .binary_store import BinaryVectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine

def get_default_store_path():
    """Get the default storage path based on the OS."""
    if os.name == 'nt':  # Windows
        base_dir = os.path.expandvars('%APPDATA%')
        return os.path.join(base_dir, 'VecStream', 'store')
    else:  # Unix-like
        home = str(Path.home())
        return os.path.join(home, '.vecstream', 'store')

class VecStreamCLI:
    def __init__(self, store_path=None):
        """Initialize CLI with binary storage.
        
        Args:
            store_path: Optional path to store the vector database.
                       If not provided, uses the default OS-specific path.
        """
        if store_path is None:
            store_path = get_default_store_path()
        
        # Ensure storage directory exists
        os.makedirs(store_path, exist_ok=True)
        
        self.store = BinaryVectorStore(store_path)
        self.index_manager = IndexManager(self.store)
        self.query_engine = QueryEngine(self.index_manager)
        self.model = None
        self.store_path = store_path
        
    def load_model(self):
        """Load the sentence transformer model if not already loaded."""
        if self.model is None:
            print("[yellow]Loading model...[/yellow]")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[green]Model loaded successfully![/green]")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to vector using the model."""
        self.load_model()
        return self.model.encode(text)

def get_cli(db_path=None):
    """Get CLI instance with optional custom db path."""
    return VecStreamCLI(db_path)

@click.group()
def main():
    """VecStream - A lightweight vector database with similarity search."""
    pass

@main.command()
@click.argument('text')
@click.argument('id')
@click.option('--db-path', help='Custom database storage path')
def add(text: str, id: str, db_path: str = None):
    """Add a text entry to the database."""
    cli = get_cli(db_path)
    vector = cli.encode_text(text)
    cli.store.add_vector(
        id=id,
        vector=vector.tolist(),
        metadata={"text": text}
    )
    print(f"[green]Added text with ID '{id}' to the database[/green]")

@main.command()
@click.argument('id')
@click.option('--db-path', help='Custom database storage path')
def get(id: str, db_path: str = None):
    """Get a text entry from the database."""
    cli = get_cli(db_path)
    try:
        vector, metadata = cli.store.get_vector_with_metadata(id)
        print(f"\n[bold cyan]Entry {id}:[/bold cyan]")
        if metadata and "text" in metadata:
            print(f"Text: {metadata['text']}")
        print(f"Vector (first 5 dimensions): {vector[:5]}...")
    except KeyError:
        print(f"[red]No entry found with ID '{id}'[/red]")

@main.command()
@click.argument('query_text')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--threshold', default=0.0, help='Minimum similarity threshold')
@click.option('--db-path', help='Custom database storage path')
def search(query_text: str, k: int, threshold: float, db_path: str = None):
    """Search for similar text entries."""
    cli = get_cli(db_path)
    query_vector = cli.encode_text(query_text)
    results = cli.store.search_similar(query_vector.tolist(), k=k)
    
    if not results:
        print("[yellow]No results found[/yellow]")
        return
    
    table = Table(title="Search Results")
    table.add_column("Score", style="cyan", justify="right")
    table.add_column("ID", style="green")
    table.add_column("Text", style="white")
    
    for doc_id, score in results:
        if score < threshold:
            continue
        _, metadata = cli.store.get_vector_with_metadata(doc_id)
        text = metadata.get("text", "N/A") if metadata else "N/A"
        table.add_row(f"{score:.4f}", doc_id, text)
    
    print(table)

@main.command()
@click.argument('id')
@click.option('--db-path', help='Custom database storage path')
def remove(id: str, db_path: str = None):
    """Remove a text entry from the database."""
    cli = get_cli(db_path)
    try:
        cli.store.remove_vector(id)
        print(f"[green]Removed entry with ID '{id}' from the database[/green]")
    except KeyError:
        print(f"[red]No entry found with ID '{id}'[/red]")

@main.command()
@click.option('--db-path', help='Custom database storage path')
def clear(db_path: str = None):
    """Clear all entries from the database."""
    cli = get_cli(db_path)
    cli.store.clear_store()
    print("[green]Database cleared successfully[/green]")

@main.command()
@click.option('--db-path', help='Custom database storage path')
def info(db_path: str = None):
    """Show database information."""
    cli = get_cli(db_path)
    vectors_size, metadata_size = cli.store.get_store_size()
    total_size = vectors_size + metadata_size
    
    table = Table(title="Database Information")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    def format_size(size_bytes):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"
    
    table.add_row("Storage Location", cli.store_path)
    table.add_row("Vectors Size", format_size(vectors_size))
    table.add_row("Metadata Size", format_size(metadata_size))
    table.add_row("Total Size", format_size(total_size))
    
    print(table)

if __name__ == "__main__":
    main() 