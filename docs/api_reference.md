# VecStream API Reference

## Core Classes

### VectorStore

The base class for vector storage and retrieval.

```python
from vecstream import VectorStore

store = VectorStore()
```

#### Methods

- `add_vector(id: str, vector: List[float]) -> None`
  - Add a vector to the store with the given ID
  - Raises `ValueError` if vector dimensions don't match

- `get_vector(id: str) -> List[float]`
  - Retrieve a vector by ID
  - Raises `KeyError` if ID not found

- `remove_vector(id: str) -> None`
  - Remove a vector from the store
  - Raises `KeyError` if ID not found

- `search_similar(query: List[float], k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]`
  - Find k most similar vectors to the query
  - Returns list of (id, similarity) tuples

### PersistentVectorStore

Extends VectorStore with persistent storage capabilities.

```python
from vecstream import PersistentVectorStore

store = PersistentVectorStore("vectors.vec")
```

#### Additional Methods

- `save() -> None`
  - Save the current state to disk

- `load() -> None`
  - Load the state from disk

### IndexManager

Manages vector indexing for efficient similarity search.

```python
from vecstream import IndexManager

index = IndexManager(vector_store)
```

#### Methods

- `create_index() -> None`
  - Create or recreate the search index

- `update_index() -> None`
  - Update the index with new vectors

- `search(query: List[float], k: int) -> List[Tuple[str, float]]`
  - Search the index for similar vectors

### QueryEngine

High-level interface for vector queries.

```python
from vecstream import QueryEngine

engine = QueryEngine(index_manager)
```

#### Methods

- `search(query: List[float], k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]`
  - Search for similar vectors with optional filtering

### VectorDBServer

Server component for distributed vector storage.

```python
from vecstream import VectorDBServer

server = VectorDBServer(host="localhost", port=5000)
server.start()
```

#### Methods

- `start() -> None`
  - Start the server

- `stop() -> None`
  - Stop the server

### VectorDBClient

Client component for connecting to a VectorDBServer.

```python
from vecstream import VectorDBClient

client = VectorDBClient(host="localhost", port=5000)
```

#### Methods

- `add_vector(id: str, vector: List[float]) -> None`
  - Add a vector through the server

- `search_similar(query: List[float], k: int = 5) -> List[Tuple[str, float]]`
  - Search for similar vectors through the server

## CLI Commands

### Add Vector
```bash
vecstream add TEXT ID [--model MODEL_NAME]
```

### Search Vectors
```bash
vecstream search TEXT [--k K] [--threshold THRESHOLD] [--model MODEL_NAME]
```

### Get Vector
```bash
vecstream get ID
```

### Remove Vector
```bash
vecstream remove ID
```

### Database Info
```bash
vecstream info
```

### Clear Database
```bash
vecstream clear
```

## Error Handling

The library uses the following exception types:

- `VectorDimensionError`: When vector dimensions don't match
- `VectorNotFoundError`: When a vector ID is not found
- `IndexNotInitializedError`: When trying to search without an index
- `ConnectionError`: When client-server communication fails

## Best Practices

1. **Memory Management**
   - Use PersistentVectorStore for large datasets
   - Call save() periodically when adding many vectors

2. **Performance Optimization**
   - Create indices before searching
   - Batch vector additions when possible
   - Use appropriate k values for search

3. **Error Handling**
   - Always catch specific exceptions
   - Implement proper cleanup in client-server mode

4. **Thread Safety**
   - VectorStore operations are not thread-safe by default
   - Use appropriate synchronization in multi-threaded environments