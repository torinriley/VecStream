# API Reference

## VectorStore

The `VectorStore` class provides methods for storing and retrieving vectors.

### Methods

#### `add(id, vector)`
Add a vector to the store with the specified ID.

#### `get(id)`
Retrieve a vector by its ID.

#### `remove(id)`
Remove a vector from the store.

## IndexManager

The `IndexManager` class handles indexing strategies for efficient vector retrieval.

### Methods

#### `create_index(metric="cosine")`
Create a new index using the specified similarity metric.

#### `update_index()`
Update the index with any new vectors.

## QueryEngine

The `QueryEngine` class performs vector similarity searches.

### Methods

#### `search(query_vector, k=10, metric="cosine")`
Search for the k most similar vectors to the query vector using the specified metric.

#### `vector_similarity(vec1, vec2, metric="cosine")`
Calculate the similarity between two vectors using the specified metric.
```