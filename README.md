# Vector Database Project

A Redis-like in-memory vector database with efficient search capabilities.

## Features

- In-memory vector storage
- Multiple indexing strategies
- Similarity search queries
- Redis-like command interface

## Installation

```bash
pip install -e .
```

## Usage

```python
from vector_db import VectorStore

# Create a vector store
store = VectorStore()

# Add vectors
store.add("doc1", [0.1, 0.2, 0.3])
store.add("doc2", [0.2, 0.3, 0.4])

# Search for similar vectors
results = store.search([0.15, 0.25, 0.35], k=5)
```

## Running the server

```bash
python scripts/run_server.py
```
s