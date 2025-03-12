# Advanced Usage Guide

## Custom Similarity Metrics

VectorRedis supports custom similarity metrics beyond the built-in cosine and euclidean options:

```python
def custom_similarity(vec1, vec2):
    # Your custom implementation
    return similarity_score

# Use your custom similarity function
query_engine.search(query_vector, k=5, metric=custom_similarity)
```

## Batch Operations

For better performance when adding multiple vectors:

```python
vectors = {
    "item1": [1.0, 0.2, 0.3],
    "item2": [0.1, 0.9, 0.5],
    "item3": [0.5, 0.5, 0.5]
}
store.batch_add(vectors)
```

## Filtering Search Results

Filter search results based on metadata:

```python
results = query_engine.search(
    query_vector=[1.0, 0.0, 0.0],
    k=10,
    filter_func=lambda id: id.startswith("product_")
)
```

## Persistence

Save and load your vector database:

```python
# Save the current state
store.save("vectors.db")

# Load from saved state
new_store = VectorStore.load("vectors.db")
```
