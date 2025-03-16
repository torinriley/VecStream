# Advanced Usage Guide

## Custom Embedding Models

VecStream supports any embedding model that produces numerical vectors. Here's how to use a custom model:

```python
from transformers import AutoModel, AutoTokenizer
from vecstream import VectorStore

class CustomEmbedding:
    def __init__(self):
        self.model = AutoModel.from_pretrained("your-model-name")
        self.tokenizer = AutoTokenizer.from_pretrained("your-model-name")
    
    def encode(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

# Use custom embeddings
embedder = CustomEmbedding()
store = VectorStore()

# Add vectors
text = "Example text"
vector = embedder.encode(text)
store.add_vector("doc1", vector)
```

## Batch Operations

For better performance when adding many vectors:

```python
from vecstream import VectorStore, IndexManager
import numpy as np

store = VectorStore()
index = IndexManager(store)

# Generate random vectors
vectors = [(f"vec_{i}", np.random.randn(384)) for i in range(10000)]

# Batch add vectors
for vec_id, vector in vectors:
    store.add_vector(vec_id, vector.tolist())

# Create index once after adding all vectors
index.create_index()
```

## Persistent Storage with Automatic Saving

```python
from vecstream import PersistentVectorStore
import time

class AutoSaveStore(PersistentVectorStore):
    def __init__(self, filepath, save_interval=300):  # 5 minutes
        super().__init__(filepath)
        self.last_save = time.time()
        self.save_interval = save_interval
    
    def add_vector(self, id, vector):
        super().add_vector(id, vector)
        if time.time() - self.last_save > self.save_interval:
            self.save()
            self.last_save = time.time()

# Use auto-saving store
store = AutoSaveStore("vectors.vec", save_interval=60)  # Save every minute
```

## Client-Server with Load Balancing

```python
from vecstream import VectorDBServer, VectorDBClient
import random

class LoadBalancedClient:
    def __init__(self, servers):
        self.servers = servers
        self.clients = [VectorDBClient(host, port) for host, port in servers]
    
    def add_vector(self, id, vector):
        # Round-robin distribution
        client_idx = hash(id) % len(self.clients)
        return self.clients[client_idx].add_vector(id, vector)
    
    def search_similar(self, query, k=5):
        # Query all servers and merge results
        all_results = []
        for client in self.clients:
            results = client.search_similar(query, k=k)
            all_results.extend(results)
        
        # Sort by similarity and return top k
        return sorted(all_results, key=lambda x: x[1], reverse=True)[:k]

# Usage
servers = [
    ("localhost", 5000),
    ("localhost", 5001),
    ("localhost", 5002)
]
client = LoadBalancedClient(servers)
```

## Custom Index Strategies

```python
from vecstream import IndexManager
import faiss

class FaissIndexManager(IndexManager):
    def create_index(self):
        dimension = next(iter(self.store.vectors.values())).shape[0]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add existing vectors
        vectors = list(self.store.vectors.values())
        if vectors:
            self.index.add(np.array(vectors))
    
    def update_index(self):
        self.create_index()  # Recreate index for simplicity
    
    def search(self, query, k):
        distances, indices = self.index.search(np.array([query]), k)
        vector_ids = list(self.store.vectors.keys())
        return [(vector_ids[i], 1/(1 + d)) for i, d in zip(indices[0], distances[0])]

# Usage
store = VectorStore()
index = FaissIndexManager(store)
```

## Thread-Safe Vector Store

```python
from vecstream import VectorStore
from threading import Lock

class ThreadSafeVectorStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.lock = Lock()
    
    def add_vector(self, id, vector):
        with self.lock:
            super().add_vector(id, vector)
    
    def get_vector(self, id):
        with self.lock:
            return super().get_vector(id)
    
    def remove_vector(self, id):
        with self.lock:
            super().remove_vector(id)
    
    def search_similar(self, query, k=5):
        with self.lock:
            return super().search_similar(query, k)

# Usage
store = ThreadSafeVectorStore()
```

## Performance Monitoring

```python
from vecstream import VectorStore
import time
import logging

class MonitoredVectorStore(VectorStore):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("vecstream")
        self.metrics = {
            "add_count": 0,
            "search_count": 0,
            "total_add_time": 0,
            "total_search_time": 0
        }
    
    def add_vector(self, id, vector):
        start = time.time()
        super().add_vector(id, vector)
        duration = time.time() - start
        
        self.metrics["add_count"] += 1
        self.metrics["total_add_time"] += duration
        self.logger.info(f"Added vector {id} in {duration:.4f}s")
    
    def search_similar(self, query, k=5):
        start = time.time()
        results = super().search_similar(query, k)
        duration = time.time() - start
        
        self.metrics["search_count"] += 1
        self.metrics["total_search_time"] += duration
        self.logger.info(f"Search completed in {duration:.4f}s")
        
        return results
    
    def get_metrics(self):
        metrics = self.metrics.copy()
        metrics["avg_add_time"] = (
            metrics["total_add_time"] / metrics["add_count"]
            if metrics["add_count"] > 0 else 0
        )
        metrics["avg_search_time"] = (
            metrics["total_search_time"] / metrics["search_count"]
            if metrics["search_count"] > 0 else 0
        )
        return metrics

# Usage
store = MonitoredVectorStore()
```

## Error Recovery

```python
from vecstream import PersistentVectorStore
import json
import os

class RecoverableStore(PersistentVectorStore):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.backup_path = filepath + ".backup"
        self.transaction_log = filepath + ".log"
    
    def add_vector(self, id, vector):
        # Log operation before executing
        with open(self.transaction_log, "a") as f:
            json.dump({"op": "add", "id": id, "vector": vector}, f)
            f.write("\n")
        
        super().add_vector(id, vector)
    
    def save(self):
        # Create backup of current state
        if os.path.exists(self.filepath):
            os.rename(self.filepath, self.backup_path)
        
        try:
            super().save()
            # Clear transaction log after successful save
            os.remove(self.transaction_log)
        except Exception as e:
            # Restore from backup if save fails
            if os.path.exists(self.backup_path):
                os.rename(self.backup_path, self.filepath)
            raise e

# Usage
store = RecoverableStore("vectors.vec")
```

## Best Practices Summary

1. **Memory Management**
   - Use batch operations for large datasets
   - Implement periodic saving for persistent storage
   - Monitor memory usage and implement cleanup strategies

2. **Performance**
   - Create indices after batch additions
   - Use appropriate batch sizes
   - Implement caching for frequent queries
   - Consider distributed storage for large datasets

3. **Reliability**
   - Implement proper error handling
   - Use transaction logs for critical operations
   - Create backup strategies
   - Monitor system performance

4. **Scalability**
   - Use load balancing for distributed setups
   - Implement proper thread safety
   - Consider sharding for large datasets
   - Monitor and optimize resource usage
