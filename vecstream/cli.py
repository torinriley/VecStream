"""
VecStream CLI - A lightweight vector database with similarity search.
"""

import click
from rich import print
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich.console import Console
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from pathlib import Path
import time

from .vector_store import VectorStore
from .binary_store import BinaryVectorStore
from .collections import CollectionManager, Collection
from .index_manager import IndexManager
from .query_engine import QueryEngine
from .hnsw_index import HNSWIndex

def get_default_store_path():
    """Get the default storage path based on the OS."""
    if os.name == 'nt':  # Windows
        base_dir = os.path.expandvars('%APPDATA%')
        return os.path.join(base_dir, 'VecStream', 'store')
    else:  # Unix-like
        home = str(Path.home())
        return os.path.join(home, '.vecstream', 'store')

class VecStreamCLI:
    def __init__(self, store_path=None, use_hnsw=True):
        """Initialize CLI with binary storage.
        
        Args:
            store_path: Optional path to store the vector database.
                       If not provided, uses the default OS-specific path.
            use_hnsw: Whether to use HNSW indexing for better performance
        """
        if store_path is None:
            store_path = get_default_store_path()
        
        # Ensure storage directory exists
        os.makedirs(store_path, exist_ok=True)
        
        # Initialize collection manager
        self.collection_manager = CollectionManager(store_path, use_hnsw=use_hnsw)
        
        # Default collection
        try:
            self.default_collection = self.collection_manager.get_collection("default")
        except KeyError:
            # Create default collection if it doesn't exist
            self.default_collection = self.collection_manager.create_collection("default")
        
        # Store reference to the binary store for backward compatibility
        self.store = self.default_collection.store
        
        # Initialize index and query engine on default collection
        self.index_manager = IndexManager(self.store, use_hnsw=use_hnsw)
        self.query_engine = QueryEngine(self.index_manager)
        
        self.model = None
        self.store_path = store_path
        self.use_hnsw = use_hnsw
        
    def load_model(self):
        """Load the sentence transformer model if not already loaded."""
        if self.model is None:
            print("[bold cyan]Loading embedding model...[/bold cyan]")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_text(self, text):
        """Embed text using the sentence transformer model."""
        self.load_model()
        return self.model.encode(text).tolist()
    
    def add_document(self, text, id, collection_name=None, metadata=None):
        """Add a document with its embedding to the store."""
        # Get or create the collection
        collection = self._get_collection(collection_name)
        
        # Generate embeddings
        embedding = self.embed_text(text)
        
        # Create default metadata if none provided
        if metadata is None:
            metadata = {"text": text, "timestamp": time.time()}
        elif isinstance(metadata, str):
            # Try to parse as JSON
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"text": text, "metadata": metadata, "timestamp": time.time()}
        else:
            # Ensure text is included in metadata
            if "text" not in metadata:
                metadata["text"] = text
        
        # Add to collection
        collection.add_vector(id, embedding, metadata)
        
        # Update index if using default collection
        if collection_name is None or collection_name == "default":
            self.index_manager.update_index()
            
        return id
    
    def search(self, query_text, collection_name=None, k=5, threshold=0.0, filter_metadata=None):
        """Search for similar documents to the query text."""
        # Get the collection
        collection = self._get_collection(collection_name)
        
        # Generate embedding for the query
        query_embedding = self.embed_text(query_text)
        
        # Perform search with filtering if needed
        if filter_metadata:
            # Convert string filter to dict if provided as string
            if isinstance(filter_metadata, str):
                try:
                    filter_metadata = json.loads(filter_metadata)
                except json.JSONDecodeError:
                    print(f"[bold red]Error: Invalid filter format.[/bold red]")
                    return []
            
            # Use HNSW index directly if available
            if collection.use_hnsw and collection.hnsw_index is not None:
                results = collection.search_similar(
                    query_embedding, 
                    k=k,
                    threshold=threshold, 
                    filter_metadata=filter_metadata
                )
            # Use the query engine otherwise
            elif collection_name is None or collection_name == "default":
                results = self.query_engine.search(
                    query_embedding, 
                    k=k, 
                    threshold=threshold,
                    filter_metadata=filter_metadata
                )
            else:
                # Fall back to standard search with manual filtering
                all_results = collection.store.search_similar(query_embedding, k=len(collection.store.vectors))
                results = []
                for id, score in all_results:
                    if score >= threshold:
                        _, meta = collection.store.get_vector_with_metadata(id)
                        if meta and self._matches_filter(meta, filter_metadata):
                            results.append((id, score))
                            if len(results) >= k:
                                break
        else:
            # No filtering needed
            if collection.use_hnsw and collection.hnsw_index is not None:
                results = collection.search_similar(query_embedding, k=k, threshold=threshold)
            elif collection_name is None or collection_name == "default":
                results = self.query_engine.search(query_embedding, k=k, threshold=threshold)
            else:
                results = collection.store.search_similar(query_embedding, k=k, threshold=threshold)
        
        return results
    
    def _matches_filter(self, metadata, filter_query):
        """Check if metadata matches the filter query."""
        for key, value in filter_query.items():
            # Handle nested keys with dot notation
            if "." in key:
                parts = key.split(".")
                current = metadata
                match = True
                
                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        match = False
                        break
                    current = current[part]
                
                if not match or parts[-1] not in current:
                    return False
                
                # Check value comparison for the last part
                if current[parts[-1]] != value:
                    return False
            else:
                # Simple key-value match
                if key not in metadata or metadata[key] != value:
                    return False
        
        return True
    
    def get_document(self, id, collection_name=None):
        """Get a document by ID."""
        collection = self._get_collection(collection_name)
        try:
            vector, metadata = collection.store.get_vector_with_metadata(id)
            return vector, metadata
        except KeyError:
            return None, None
    
    def remove_document(self, id, collection_name=None):
        """Remove a document by ID."""
        collection = self._get_collection(collection_name)
        try:
            collection.remove_vector(id)
            # Update index if using default collection
            if collection_name is None or collection_name == "default":
                self.index_manager.update_index()
            return True
        except KeyError:
            return False
    
    def _get_collection(self, collection_name=None):
        """Get a collection by name, falling back to default."""
        if collection_name is None:
            return self.default_collection
        
        try:
            return self.collection_manager.get_collection(collection_name)
        except KeyError:
            # Create collection if it doesn't exist
            return self.collection_manager.create_collection(collection_name, use_hnsw=self.use_hnsw)
    
    def list_collections(self):
        """List all available collections."""
        return self.collection_manager.list_collections()
    
    def get_collection_stats(self, collection_name):
        """Get statistics for a collection."""
        try:
            return self.collection_manager.get_collection_stats(collection_name)
        except KeyError:
            return None
    
    def delete_collection(self, collection_name):
        """Delete a collection."""
        if collection_name == "default":
            print("[bold red]Cannot delete default collection.[/bold red]")
            return False
        
        try:
            self.collection_manager.delete_collection(collection_name)
            return True
        except KeyError:
            return False


@click.group()
@click.option('--db-path', help='Custom path for vector database storage.')
@click.option('--use-hnsw/--no-hnsw', default=True, help='Whether to use HNSW indexing for better performance.')
@click.pass_context
def cli(ctx, db_path, use_hnsw):
    """VecStream - A lightweight vector database with similarity search."""
    ctx.ensure_object(dict)
    ctx.obj['cli'] = VecStreamCLI(db_path, use_hnsw=use_hnsw)


@cli.command()
@click.argument('text')
@click.argument('id')
@click.option('--collection', '-c', help='Collection to add the document to.')
@click.option('--metadata', '-m', help='JSON metadata to associate with the document.')
@click.pass_context
def add(ctx, text, id, collection, metadata):
    """Add a document with TEXT and ID to the database."""
    cli_instance = ctx.obj['cli']
    
    try:
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                print(f"[bold red]Error: Invalid JSON metadata.[/bold red]")
                return
        else:
            metadata_dict = {"text": text, "timestamp": time.time()}
        
        cli_instance.add_document(text, id, collection, metadata_dict)
        collection_info = f" in collection '{collection}'" if collection else ""
        print(f"[bold green]Added document with ID '{id}'{collection_info}[/bold green]")
    except Exception as e:
        print(f"[bold red]Error adding document: {str(e)}[/bold red]")


@cli.command()
@click.argument('query')
@click.option('--k', default=5, help='Number of results to return.')
@click.option('--threshold', default=0.0, help='Minimum similarity threshold (0-1).')
@click.option('--collection', '-c', help='Collection to search in.')
@click.option('--filter', '-f', help='JSON metadata filter to apply to search results.')
@click.pass_context
def search(ctx, query, k, threshold, collection, filter):
    """Search for documents similar to QUERY."""
    cli_instance = ctx.obj['cli']
    
    try:
        filter_dict = None
        if filter:
            try:
                filter_dict = json.loads(filter)
            except json.JSONDecodeError:
                print(f"[bold red]Error: Invalid JSON filter.[/bold red]")
                return
        
        results = cli_instance.search(query, collection, k, threshold, filter_dict)
    
    if not results:
            print("[yellow]No matching documents found.[/yellow]")
        return
    
        console = Console()
        collection_info = f" in '{collection}'" if collection else ""
        table = Table(title=f"Search Results{collection_info}", show_header=True)
        table.add_column("ID", style="cyan")
        table.add_column("Similarity", style="green")
    table.add_column("Text", style="white")
    
        for id, similarity in results:
            _, metadata = cli_instance.get_document(id, collection)
        text = metadata.get("text", "N/A") if metadata else "N/A"
            # Truncate text if too long
            if len(text) > 60:
                text = text[:57] + "..."
            table.add_row(id, f"{similarity:.4f}", text)
        
        console.print(table)
    except Exception as e:
        print(f"[bold red]Error searching: {str(e)}[/bold red]")


@cli.command()
@click.argument('id')
@click.option('--collection', '-c', help='Collection to get the document from.')
@click.pass_context
def get(ctx, id, collection):
    """Get a document by ID."""
    cli_instance = ctx.obj['cli']
    
    try:
        _, metadata = cli_instance.get_document(id, collection)
        
        if metadata is None:
            collection_info = f" in collection '{collection}'" if collection else ""
            print(f"[yellow]Document with ID '{id}'{collection_info} not found.[/yellow]")
            return
        
        console = Console()
        collection_info = f" from '{collection}'" if collection else ""
        panel = Panel(
            json.dumps(metadata, indent=2),
            title=f"Document '{id}'{collection_info}",
            border_style="green"
        )
        console.print(panel)
    except Exception as e:
        print(f"[bold red]Error retrieving document: {str(e)}[/bold red]")


@cli.command()
@click.argument('id')
@click.option('--collection', '-c', help='Collection to remove the document from.')
@click.pass_context
def remove(ctx, id, collection):
    """Remove a document by ID."""
    cli_instance = ctx.obj['cli']
    
    try:
        success = cli_instance.remove_document(id, collection)
        
        if success:
            collection_info = f" from collection '{collection}'" if collection else ""
            print(f"[bold green]Removed document with ID '{id}'{collection_info}[/bold green]")
        else:
            collection_info = f" in collection '{collection}'" if collection else ""
            print(f"[yellow]Document with ID '{id}'{collection_info} not found.[/yellow]")
    except Exception as e:
        print(f"[bold red]Error removing document: {str(e)}[/bold red]")


@cli.command()
@click.option('--collection', '-c', help='Collection to display info for (omit for all collections).')
@click.pass_context
def info(ctx, collection):
    """Display information about the vector database."""
    cli_instance = ctx.obj['cli']
    
    try:
        # If collection specified, show collection info
        if collection:
            stats = cli_instance.get_collection_stats(collection)
            if not stats:
                print(f"[yellow]Collection '{collection}' not found.[/yellow]")
                return
            
            console = Console()
            table = Table(title=f"Collection: {collection}", show_header=True)
            table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
            for key, value in stats.items():
                if key in ["vector_count", "dimension", "using_hnsw"]:
                    table.add_row(key, str(value))
            
            # Format sizes
            vec_size_mb = stats.get("vectors_size_bytes", 0) / (1024 * 1024)
            meta_size_mb = stats.get("metadata_size_bytes", 0) / (1024 * 1024)
            total_size_mb = stats.get("total_size_bytes", 0) / (1024 * 1024)
            
            table.add_row("vectors_size", f"{vec_size_mb:.2f} MB")
            table.add_row("metadata_size", f"{meta_size_mb:.2f} MB")
            table.add_row("total_size", f"{total_size_mb:.2f} MB")
            
            console.print(table)
        
        # Show list of collections
        else:
            collections = cli_instance.list_collections()
            
            console = Console()
            table = Table(title="VecStream Collections", show_header=True)
            table.add_column("Collection", style="cyan")
            table.add_column("Vectors", style="green")
            table.add_column("Size", style="yellow")
            table.add_column("HNSW Index", style="blue")
            
            for coll_name in collections:
                stats = cli_instance.get_collection_stats(coll_name)
                if stats:
                    vector_count = stats.get("vector_count", 0)
                    total_size_mb = stats.get("total_size_bytes", 0) / (1024 * 1024)
                    using_hnsw = "Yes" if stats.get("using_hnsw", False) else "No"
                    
                    table.add_row(
                        coll_name,
                        str(vector_count),
                        f"{total_size_mb:.2f} MB", 
                        using_hnsw
                    )
            
            console.print(table)
            
            # Display storage path
            print(f"\n[bold]Storage Location:[/bold] {cli_instance.store_path}")
    except Exception as e:
        print(f"[bold red]Error displaying information: {str(e)}[/bold red]")


@cli.command()
@click.argument('name')
@click.option('--use-hnsw/--no-hnsw', default=True, help='Whether to use HNSW indexing.')
@click.pass_context
def create_collection(ctx, name, use_hnsw):
    """Create a new collection."""
    cli_instance = ctx.obj['cli']
    
    try:
        collection = cli_instance.collection_manager.create_collection(name, use_hnsw=use_hnsw)
        print(f"[bold green]Created collection '{name}'[/bold green]")
    except ValueError as e:
        print(f"[bold red]{str(e)}[/bold red]")
    except Exception as e:
        print(f"[bold red]Error creating collection: {str(e)}[/bold red]")


@cli.command()
@click.argument('name')
@click.pass_context
def delete_collection(ctx, name):
    """Delete a collection."""
    cli_instance = ctx.obj['cli']
    
    try:
        if name == "default":
            print("[bold red]Cannot delete default collection.[/bold red]")
            return
        
        success = cli_instance.delete_collection(name)
        if success:
            print(f"[bold green]Deleted collection '{name}'[/bold green]")
        else:
            print(f"[yellow]Collection '{name}' not found.[/yellow]")
    except Exception as e:
        print(f"[bold red]Error deleting collection: {str(e)}[/bold red]")


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main() 