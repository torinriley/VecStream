"""
VectorStore class for managing vector storage and similarity search.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

class VectorStore:
    """A class for storing and searching vectors."""

    def __init__(self):
        """Initialize an empty vector store."""
        self.vectors: Dict[str, np.ndarray] = {}
        self.dimension: Optional[int] = None

    def add_vector(self, id: str, vector: List[float]) -> None:
        """Add a vector to the store.
        
        Args:
            id: Unique identifier for the vector
            vector: List of float values representing the vector
        
        Raises:
            ValueError: If vector dimensions don't match existing vectors
        """
        vector_array = np.array(vector, dtype=np.float32)
        
        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} does not match store dimension {self.dimension}")
        
        self.vectors[id] = vector_array

    def get_vector(self, id: str) -> List[float]:
        """Retrieve a vector by ID.
        
        Args:
            id: The vector's identifier
            
        Returns:
            The vector as a list of floats
            
        Raises:
            KeyError: If the ID is not found
        """
        if id not in self.vectors:
            raise KeyError(f"Vector with ID {id} not found")
        return self.vectors[id].tolist()

    def remove_vector(self, id: str) -> None:
        """Remove a vector from the store.
        
        Args:
            id: The vector's identifier
            
        Raises:
            KeyError: If the ID is not found
        """
        if id not in self.vectors:
            raise KeyError(f"Vector with ID {id} not found")
        del self.vectors[id]

    def search_similar(self, query: List[float], k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Find k most similar vectors to the query vector.
        
        Args:
            query: Query vector to find similar vectors for
            k: Number of similar vectors to return
            threshold: Minimum similarity score (0 to 1) for results
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity (highest first)
        """
        if not self.vectors:
            return []

        query_array = np.array(query, dtype=np.float32)
        if len(query_array) != self.dimension:
            raise ValueError(f"Query dimension {len(query_array)} does not match store dimension {self.dimension}")

        # Normalize query vector
        query_norm = np.linalg.norm(query_array)
        if query_norm > 0:
            query_array = query_array / query_norm

        similarities = []
        for id, vec in self.vectors.items():
            # Normalize stored vector
            vec_norm = np.linalg.norm(vec)
            if vec_norm > 0:
                vec = vec / vec_norm
            
            # Compute cosine similarity
            similarity = np.dot(query_array, vec)
            if similarity >= threshold:
                similarities.append((id, float(similarity)))

        # Sort by similarity (highest first) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

