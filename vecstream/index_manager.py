"""
IndexManager class for managing vector indices for efficient similarity search.
"""

import numpy as np
from typing import List, Tuple, Optional
from .vector_store import VectorStore

class IndexManager:
    """A class for managing vector indices for efficient similarity search."""

    def __init__(self, store: VectorStore):
        """Initialize the index manager.
        
        Args:
            store: VectorStore instance to manage indices for
        """
        self.store = store
        self.index = None
        self.indexed_ids = []

    def create_index(self) -> None:
        """Create or recreate the search index."""
        if not self.store.vectors:
            return

        # Get all vectors and their IDs
        vectors = []
        self.indexed_ids = []
        for id, vector in self.store.vectors.items():
            vectors.append(vector)
            self.indexed_ids.append(id)

        # Convert to numpy array and normalize
        vectors_array = np.array(vectors)
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.index = vectors_array / norms

    def update_index(self) -> None:
        """Update the index with new vectors."""
        if self.index is None:
            self.create_index()
            return

        # Check for new vectors
        current_ids = set(self.store.vectors.keys())
        indexed_ids = set(self.indexed_ids)
        
        if current_ids != indexed_ids:
            self.create_index()  # Recreate index if vectors have changed

    def search(self, query: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search the index for similar vectors.
        
        Args:
            query: Query vector to find similar vectors for
            k: Number of similar vectors to return
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity (highest first)
        """
        if self.index is None or not self.indexed_ids:
            return []

        # Normalize query vector
        query_array = np.array(query, dtype=np.float32)
        query_norm = np.linalg.norm(query_array)
        if query_norm > 0:
            query_array = query_array / query_norm

        # Compute similarities
        similarities = np.dot(self.index, query_array)
        
        # Get top k results
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            id = self.indexed_ids[idx]
            similarity = float(similarities[idx])
            results.append((id, similarity))
            
        return results

