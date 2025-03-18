# VecStream

[![Tests](https://github.com/torinriley/VecStream/actions/workflows/tests.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/tests.yml)
[![Benchmarks](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml)
[![PyPI version](https://badge.fury.io/py/vecstream.svg)](https://badge.fury.io/py/vecstream)
[![Python versions](https://img.shields.io/pypi/pyversions/vecstream.svg)](https://pypi.org/project/vecstream/)
[![License](https://img.shields.io/github/license/torinriley/VecStream.svg)](https://github.com/torinriley/VecStream/blob/main/LICENSE)

A lightweight, efficient vector database with similarity search capabilities, designed for machine learning and AI applications.

## Features

- Fast similarity search using optimized indexing
- HNSW indexing for significantly improved search performance
- Vector collections/namespaces for organizing different types of embeddings
- Metadata filtering for fine-grained search control
- Efficient binary storage format for vectors and metadata
- Automatic text embedding with sentence-transformers
- Rich command-line interface with beautiful output
- Cross-platform support (Windows, macOS, Linux)
- Customizable storage locations
- Metadata support for enhanced document management
- Built-in text similarity search

## Installation

```bash
pip install vecstream
```

## Quick Start

### Using the CLI

```bash
# Add a document
vecstream add "Machine learning is transforming technology" doc1

# Search for similar documents
vecstream search "AI and machine learning" --k 3

# Search with metadata filtering
vecstream search "cloud computing" --filter '{"category": "ai", "year": 2023}'

# Get document by ID
vecstream get doc1

# View database information
vecstream info

# Create and use a collection
vecstream create_collection research
vecstream add "Neural networks research" doc2 --collection research

# Use custom storage location
vecstream add "Custom storage test" doc3 --db-path "./my_vectors"

# Remove a document
vecstream remove doc1
```

### Using the Python API

```python
from vecstream.collections import CollectionManager
from vecstream.binary_store import BinaryVectorStore

# Using collections for different vector types
manager = CollectionManager("./vector_db")
research_collection = manager.create_collection("research")
products_collection = manager.create_collection("products")

# Add vectors with metadata to collections
research_collection.add_vector(
    id="paper1",
    vector=[1.0, 0.0, 0.0],
    metadata={"topic": "AI", "year": 2023, "author": "Smith"}
)

# Search with metadata filtering
results = research_collection.search_similar(
    query=[1.0, 0.0, 0.0],
    k=5,
    filter_metadata={"year": 2023, "topic": "AI"}
)

# Basic binary store usage (compatible with earlier versions)
store = BinaryVectorStore("./vector_db")

# Add vectors with metadata
store.add_vector(
    id="doc1",
    vector=[1.0, 0.0, 0.0],
    metadata={"text": "Example document", "tags": ["test"]}
)

# Search similar vectors
results = store.search_similar([1.0, 0.0, 0.0], k=5)

# Get vector with metadata
vector, metadata = store.get_vector_with_metadata("doc1")
```

## Storage Locations

By default, VecStream stores its data in:
- Windows: `%APPDATA%/VecStream/store/`
- macOS/Linux: `~/.vecstream/store/`

You can specify a custom storage location using the `--db-path` option in CLI commands or by passing the path to `CollectionManager` or `BinaryVectorStore`.

## Storage Format

VecStream uses an efficient binary storage format:
- Vectors: NumPy `.npy` format for fast access
- Metadata: JSON format for flexibility
- Automatic compression and optimization
- Collections organized in subdirectories

## CLI Features

The command-line interface provides:
- **Vector Management**: Add, get, update and remove vectors with `add`, `get`, and `remove` commands
- **Similarity Search**: Fast vector search with `search` command with adjustable k-nearest neighbors
- **HNSW Indexing**: Significantly faster search performance for large datasets (up to 100x faster)
- **Collections**: Organize vectors by type with `collection create`, `collection list`, and other commands
- **Metadata Filtering**: Filter search results with `--filter '{"key": "value"}'` syntax
- **Nested Filters**: Support for dot notation in filters like `--filter '{"details.color": "red"}'`
- **Beautiful UI**: Rich, colored output and progress indicators for long operations
- **Database Stats**: View detailed database information with `info` command
- **Custom Storage**: Specify storage locations with `--db-path` option

## Python API

The Python API offers:
- **HNSW Indexing**: Fast approximate nearest-neighbor search with customizable parameters:
  ```python
  from vecstream.hnsw_index import HNSWIndex
  index = HNSWIndex(dim=128, M=16, ef_construction=200)
  ```
- **Collections**: Organize vectors with the CollectionManager:
  ```python
  from vecstream.collections import CollectionManager
  manager = CollectionManager("./vector_db", use_hnsw=True)
  collection = manager.create_collection("images")
  ```
- **Metadata Filtering**: Fine-grained search control:
  ```python
  results = collection.search_similar(query, filter_metadata={"category": "electronics"})
  ```
- **Nested Filtering**: Access nested properties with dot notation:
  ```python
  results = collection.search_similar(query, filter_metadata={"details.color": "black"})
  ```
- **Binary Storage**: Efficient serialization for large datasets:
  ```python
  from vecstream.binary_store import BinaryVectorStore
  store = BinaryVectorStore("./vector_db")
  ```
- **Vector Operations**: Direct access to similarity calculations, normalization, and more
- **Type Safety**: Strong typing and error handling with descriptive exceptions

## Requirements

- Python 3.8 or higher
- NumPy
- SciPy
- sentence-transformers
- Rich (for CLI)
- Click (for CLI)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version History

- 0.3.0 (2024-03-XX)
  - Added HNSW indexing for faster similarity search
  - Added collections/namespaces for organizing vectors
  - Added metadata filtering for search results
  - Improved CLI with collection management commands
  - Performance optimizations

- 0.2.0 (2024-03-XX)
  - Added binary vector store
  - Improved persistent storage
  - Enhanced CLI functionality
  - Added metadata support

- 0.1.0 (2024-03-XX)
  - Initial release
  - Basic vector storage and search functionality
  - CLI interface
  - Client-server architecture



# Documentation

| Document | Description | Link |
|----------|-------------|------|
| API Reference | Complete reference of VecStream's classes, methods, and CLI commands | [API Reference](https://github.com/torinriley/VecStream/blob/main/docs/api_reference.md) |
| Advanced Usage | Detailed examples and best practices for using VecStream | [Advanced Usage](https://github.com/torinriley/VecStream/blob/main/docs/advanced_usage.md) |

## Key Features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| HNSW Indexing | Fast approximate nearest neighbor search for large datasets | [API Reference](https://github.com/torinriley/VecStream/blob/main/docs/api_reference.md#hnswindex), [Usage Examples](https://github.com/torinriley/VecStream/blob/main/docs/advanced_usage.md#hnsw-indexing-for-faster-search) |
| Collections | Organize vectors with metadata for better organization | [API Reference](https://github.com/torinriley/VecStream/blob/main/docs/api_reference.md#collection), [Usage Examples](https://github.com/torinriley/VecStream/blob/main/docs/advanced_usage.md#working-with-collections) |
| Metadata Filtering | Filter search results using metadata properties | [API Reference](https://github.com/torinriley/VecStream/blob/main/docs/api_reference.md#metadata-filtering), [Usage Examples](https://github.com/torinriley/VecStream/blob/main/docs/advanced_usage.md#advanced-metadata-filtering) |
| Binary Storage | Efficient storage format for large vector datasets | [API Reference](https://github.com/torinriley/VecStream/blob/main/docs/api_reference.md#binaryvectorstore), [Usage Examples](https://github.com/torinriley/VecStream/blob/main/docs/advanced_usage.md#binary-storage-for-efficiency) |