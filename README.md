# VecStream

[![Tests](https://github.com/torinriley/VecStream/actions/workflows/tests.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/tests.yml)
[![Benchmarks](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml)
[![PyPI version](https://badge.fury.io/py/vecstream.svg)](https://badge.fury.io/py/vecstream)
[![Python versions](https://img.shields.io/pypi/pyversions/vecstream.svg)](https://pypi.org/project/vecstream/)
[![License](https://img.shields.io/github/license/torinriley/VecStream.svg)](https://github.com/torinriley/VecStream/blob/main/LICENSE)

A lightweight, efficient vector database with similarity search capabilities, designed for machine learning and AI applications.

## Features

- Fast similarity search using optimized indexing
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

# Get document by ID
vecstream get doc1

# View database information
vecstream info

# Use custom storage location
vecstream add "Custom storage test" doc2 --db-path "./my_vectors"

# Remove a document
vecstream remove doc1
```

### Using the Python API

```python
from vecstream.binary_store import BinaryVectorStore

# Create a binary vector store
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

You can specify a custom storage location using the `--db-path` option in CLI commands or by passing the path to `BinaryVectorStore`.

## Storage Format

VecStream uses an efficient binary storage format:
- Vectors: NumPy `.npy` format for fast access
- Metadata: JSON format for flexibility
- Automatic compression and optimization

## CLI Features

The command-line interface provides:
- Beautiful, colored output using Rich
- Progress indicators for long operations
- Detailed database information
- Similarity scores in search results
- Customizable search parameters
- Error handling and user feedback

## Python API

The Python API offers:
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

- 0.1.1 (2024-03-XX)
  - Fixed index initialization in IndexManager
  - Added specific version requirements for torch and torchvision
  - Improved dependency compatibility
  - Fixed CLI import issues

- 0.1.0 (2024-03-XX)
  - Initial release
  - Basic vector storage and search functionality
  - CLI interface
  - Client-server architecture
