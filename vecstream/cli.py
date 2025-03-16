import click
from rich import print
from rich.table import Table
from rich.progress import track
from sentence_transformers import SentenceTransformer
import numpy as np

from .vector_store import VectorStore
from .index_manager import IndexManager
from .query_engine import QueryEngine
from .persistent_store import PersistentVectorStore

class VecStreamCLI:
    def __init__(self):
        self.store = VectorStore()
        self.index_manager = IndexManager(self.store)
        self.query_engine = QueryEngine(self.index_manager)
        self.model = None
        
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

cli = VecStreamCLI()

@click.group()
def main():
    """VecStream - A lightweight vector database with similarity search."""
    pass

@main.command()
@click.argument('text')
@click.argument('id')
def add(text: str, id: str):
    """Add a text entry to the database."""
    vector = cli.encode_text(text)
    cli.store.add(id, vector)
    print(f"[green]Added text with ID '{id}' to the database[/green]")

@main.command()
@click.argument('id')
def get(id: str):
    """Get a vector by its ID."""
    vector = cli.store.get(id)
    if vector is None:
        print(f"[red]No vector found with ID '{id}'[/red]")
        return
    
    table = Table(title=f"Vector {id}")
    table.add_column("Dimension", justify="right", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    for i, value in enumerate(vector[:10]):  # Show first 10 dimensions
        table.add_row(str(i), f"{value:.6f}")
    
    if len(vector) > 10:
        table.add_row("...", "...")
    
    print(table)
    print(f"\nVector shape: {vector.shape}")

@main.command()
@click.argument('query_text')
@click.option('--k', default=5, help='Number of results to return')
@click.option('--threshold', default=0.0, help='Minimum similarity threshold')
def search(query_text: str, k: int, threshold: float):
    """Search for similar vectors."""
    query_vector = cli.encode_text(query_text)
    results = cli.query_engine.search(query_vector, k=k)
    
    table = Table(title="Search Results")
    table.add_column("ID", style="cyan")
    table.add_column("Similarity", justify="right", style="green")
    
    for id, similarity in results:
        if similarity >= threshold:
            table.add_row(id, f"{similarity:.4f}")
    
    print(table)

@main.command()
@click.argument('id')
def remove(id: str):
    """Remove a vector from the database."""
    if cli.store.remove(id):
        print(f"[green]Removed vector with ID '{id}'[/green]")
    else:
        print(f"[red]No vector found with ID '{id}'[/red]")

@main.command()
def clear():
    """Clear all vectors from the database."""
    cli.store.clear()
    print("[green]Database cleared successfully[/green]")

@main.command()
def info():
    """Display information about the database."""
    table = Table(title="VecStream Database Info")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    table.add_row("Number of vectors", str(len(cli.store)))
    if len(cli.store) > 0:
        sample_vector = next(iter(cli.store.vectors.values()))
        table.add_row("Vector dimensions", str(sample_vector.shape[0]))
    
    print(table)

if __name__ == '__main__':
    main() 