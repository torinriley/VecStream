# VecStream

[![Tests](https://github.com/torinriley/VecStream/actions/workflows/tests.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/tests.yml)
[![Benchmarks](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/torinriley/VecStream/actions/workflows/benchmarks.yml)
[![PyPI version](https://badge.fury.io/py/vecstream.svg)](https://badge.fury.io/py/vecstream)
[![Python versions](https://img.shields.io/pypi/pyversions/vecstream.svg)](https://pypi.org/project/vecstream/)
[![License](https://img.shields.io/github/license/torinriley/VecStream.svg)](https://github.com/torinriley/VecStream/blob/main/LICENSE)

A lightweight, efficient vector database with similarity search capabilities, designed for machine learning and AI applications.

## Features

- Fast similarity search using optimized indexing
- In-memory and persistent storage options
- Client-server architecture for distributed use
- Command-line interface for easy management
- Built-in support for text embeddings via sentence-transformers
- Efficient vector operations with NumPy backend

## Installation

```bash
pip install vecstream
```

## Quick Start

### Using the CLI

```bash
# Add a vector
vecstream add "This is a test sentence" test1

# Search for similar vectors
vecstream search "test sentence" --k 2

# Get database info
vecstream info
```

### Using the Python API

```python
from vecstream import VectorStore

# Create a vector store
store = VectorStore()

# Add vectors
store.add_vector("vec1", [1.0, 0.0, 0.0])
store.add_vector("vec2", [0.0, 1.0, 0.0])

# Search for similar vectors
results = store.search_similar([1.0, 0.1, 0.1], k=2)
for vec_id, similarity in results:
    print(f"{vec_id}: {similarity}")
```

### Using Persistent Storage

```python
from vecstream import PersistentVectorStore

# Create a persistent store
store = PersistentVectorStore("vectors.vec")

# Add and search vectors as with VectorStore
store.add_vector("vec1", [1.0, 0.0, 0.0])
results = store.search_similar([1.0, 0.1, 0.1], k=2)
```

### Using Client-Server Mode

```python
from vecstream import VectorDBServer, VectorDBClient

# Start the server
server = VectorDBServer()
server.start()

# Connect a client
client = VectorDBClient()
client.add_vector("vec1", [1.0, 0.0, 0.0])
results = client.search_similar([1.0, 0.1, 0.1], k=2)
```

## Performance

VecStream is optimized for both speed and memory efficiency. Here are some key performance metrics:

- Vector addition: ~100,000 vectors/second (384-dimensional)
- Similarity search: ~1,000 queries/second (100K vector database)
- Memory usage: ~400 bytes per 384-dimensional vector

For detailed benchmarks, run:
```bash
python -m tests.benchmarks.benchmark_vector_operations
```

## Development

### Setup

1. Clone the repository:
```bash
git clone https://github.com/torinetheridge/vecstream.git
cd vecstream
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit
pytest tests/integration

# Run benchmarks
python tests/benchmarks/benchmark_vector_operations.py
```

### Project Structure

```
vecstream/
├── vecstream/
│   ├── __init__.py
│   ├── cli.py
│   ├── vector_store.py
│   ├── persistent_store.py
│   ├── index_manager.py
│   ├── query_engine.py
│   ├── server.py
│   └── client.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── docs/
│   ├── advanced_usage.md
│   └── api_reference.md
├── README.md
├── LICENSE
├── setup.py
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

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
