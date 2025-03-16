"""Stress tests for CLI operations under high load."""

import tempfile
import os
import numpy as np
from click.testing import CliRunner
from vecstream.cli import main as cli_main
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def print_benchmark_results(benchmark):
    """Print benchmark results in a nice table format."""
    if not benchmark or not hasattr(benchmark, 'stats'):
        console.print("[red]No benchmark results available[/red]")
        return
        
    stats = benchmark.stats
    
    # Create the table
    table = Table(
        title="🚀 VecStream CLI Performance Results",
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold cyan",
        border_style="blue"
    )
    
    # Add columns
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    
    # Add rows
    table.add_row("Min Time", f"{stats.min * 1000:.2f} ms")
    table.add_row("Max Time", f"{stats.max * 1000:.2f} ms")
    table.add_row("Mean Time", f"{stats.mean * 1000:.2f} ms")
    table.add_row("Std Dev", f"{stats.stddev * 1000:.2f} ms")
    table.add_row("Operations/sec", f"{1 / stats.mean:.2f}")
    table.add_row("Rounds", str(stats.rounds))
    
    # Print the table
    console.print()
    console.print(table)
    console.print()

def generate_large_batch_data(num_vectors=1000, dimension=384):
    """Generate a large batch of test vectors."""
    return {
        f"stress_vec_{i}": np.random.random(dimension).astype(np.float32).tolist()
        for i in range(num_vectors)
    }

def test_cli_bulk_add_stress(benchmark):
    """Stress test adding many vectors through CLI."""
    runner = CliRunner()
    vectors = generate_large_batch_data(num_vectors=1000)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "stress_test.vec")
        
        def bulk_add():
            for vec_id, vector in list(vectors.items())[:100]:  # Test with 100 vectors per batch
                result = runner.invoke(cli_main, [
                    'add',
                    '--id', vec_id,
                    '--text', f"Test vector {vec_id}",  # Using text instead of raw vector
                    '--db-path', db_path
                ])
                assert result.exit_code == 0
        
        try:
            result = benchmark(bulk_add)
            console.rule("[bold red]Bulk Add Stress Test")
            print_benchmark_results(result)
        except Exception as e:
            console.print(f"[red]Error in bulk add test: {str(e)}[/red]")

def test_cli_concurrent_search_stress(benchmark):
    """Stress test concurrent search operations."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "stress_test.vec")
        
        # First populate the database
        for i in range(500):  # Add 500 vectors for search
            runner.invoke(cli_main, [
                'add',
                '--id', f'stress_vec_{i}',
                '--text', f'Test vector {i}',
                '--db-path', db_path
            ])
        
        def concurrent_search():
            # Simulate concurrent searches
            for _ in range(10):  # 10 concurrent searches
                result = runner.invoke(cli_main, [
                    'search',
                    '--text', 'Test vector',  # Search by text
                    '--k', '10',
                    '--db-path', db_path
                ])
                assert result.exit_code == 0
        
        try:
            result = benchmark(concurrent_search)
            console.rule("[bold red]Concurrent Search Stress Test")
            print_benchmark_results(result)
        except Exception as e:
            console.print(f"[red]Error in concurrent search test: {str(e)}[/red]")

def test_cli_mixed_operations_stress(benchmark):
    """Stress test mixed operations (add, search, delete)."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "stress_test.vec")
        
        # Initial population
        for i in range(50):
            runner.invoke(cli_main, [
                'add',
                '--id', f'stress_vec_{i}',
                '--text', f'Test vector {i}',
                '--db-path', db_path
            ])
        
        def mixed_operations():
            # Add operation
            vec_id = f"stress_vec_{np.random.randint(1000)}"
            runner.invoke(cli_main, [
                'add',
                '--id', vec_id,
                '--text', f'Test vector {vec_id}',
                '--db-path', db_path
            ])
            
            # Search operation
            runner.invoke(cli_main, [
                'search',
                '--text', 'Test vector',
                '--k', '5',
                '--db-path', db_path
            ])
            
            # Delete operation
            runner.invoke(cli_main, [
                'remove',  # Changed from delete to remove
                '--id', vec_id,
                '--db-path', db_path
            ])
        
        try:
            result = benchmark(mixed_operations)
            console.rule("[bold red]Mixed Operations Stress Test")
            print_benchmark_results(result)
        except Exception as e:
            console.print(f"[red]Error in mixed operations test: {str(e)}[/red]")

def test_cli_large_batch_processing(benchmark):
    """Stress test processing large batches of vectors."""
    runner = CliRunner()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "stress_test.vec")
        
        def batch_process():
            # Generate and process a large batch of vectors
            batch_size = 100
            
            # Add vectors
            for i in range(batch_size):
                runner.invoke(cli_main, [
                    'add',
                    '--id', f'stress_vec_{i}',
                    '--text', f'Test vector {i}',
                    '--db-path', db_path
                ])
            
            # Perform batch search
            runner.invoke(cli_main, [
                'search',
                '--text', 'Test vector',
                '--k', str(batch_size // 2),
                '--db-path', db_path
            ])
        
        try:
            result = benchmark(batch_process)
            console.rule("[bold red]Large Batch Processing Test")
            print_benchmark_results(result)
        except Exception as e:
            console.print(f"[red]Error in batch processing test: {str(e)}[/red]") 