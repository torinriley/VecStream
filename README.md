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

## Advanced Features

### HNSW Indexing

VecStream now implements HNSW (Hierarchical Navigable Small World) indexing for dramatically faster similarity search, especially with large datasets:

```python
from vecstream.hnsw_index import HNSWIndex
from vecstream.collections import CollectionManager

# Collections use HNSW by default
manager = CollectionManager("./vector_db", use_hnsw=True)
collection = manager.create_collection(
    "products",
    hnsw_params={
        "M": 16,                 # Max connections per node
        "ef_construction": 200   # Build-time exploration factor
    }
)

# Add vectors
collection.add_vector("product1", [0.5, 0.2, 0.8], {"name": "Desk Lamp"})

# Search with custom ef_search parameter
results = collection.search_similar(
    [0.5, 0.2, 0.8], 
    k=10,
    ef_search=100  # Search-time exploration factor
)
```

### Collections/Namespaces

Organize your vectors into separate collections:

```python
from vecstream.collections import CollectionManager

# Create a collection manager
manager = CollectionManager("./vector_db")

# Create different collections
images = manager.create_collection("images")
texts = manager.create_collection("texts")
products = manager.create_collection("products")

# Add vectors to specific collections
images.add_vector("img1", [...], {"format": "jpg", "size": "1024x768"})
texts.add_vector("text1", [...], {"language": "en", "word_count": 500})

# List all collections
collection_names = manager.list_collections()

# Get collection statistics
stats = manager.get_collection_stats("images")
```

### Metadata Filtering

Filter search results based on metadata properties:

```python
# Search with simple metadata filter
results = collection.search_similar(
    query_vector,
    k=5,
    filter_metadata={"category": "electronics", "in_stock": True}
)

# Search with nested metadata filter using dot notation
results = collection.search_similar(
    query_vector,
    k=5,
    filter_metadata={
        "details.color": "blue",
        "ratings.average": 4.5
    }
)
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
- Beautiful, colored output using Rich
- Progress indicators for long operations
- Detailed database information
- Similarity scores in search results
- Customizable search parameters
- Collection management
- Metadata filtering
- Error handling and user feedback

## Python API

The Python API offers:
- Collections for organizing vectors
- HNSW indexing for fast search
- Metadata filtering capabilities 
- Direct access to vector operations
- Metadata management
- Custom storage locations
- Efficient binary serialization
- Rich search capabilities
- Error handling and type safety

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
