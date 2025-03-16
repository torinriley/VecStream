"""
QueryEngine class for high-level vector query operations.
"""

from typing import List, Tuple, Optional
from .index_manager import IndexManager
import numpy as np
from scipy.spatial import distance

class QueryEngine:
    """A class for high-level vector query operations."""

    def __init__(self, index_manager: IndexManager):
        """Initialize the query engine.
        
        Args:
            index_manager: IndexManager instance to use for queries
        """
        self.index_manager = index_manager

    def search(self, query: List[float], k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar vectors with optional filtering.
        
        Args:
            query: Query vector to find similar vectors for
            k: Number of similar vectors to return
            threshold: Minimum similarity score (0 to 1) for results
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity (highest first)
        """
        # Update index before searching
        self.index_manager.update_index()
        
        # Get results from index
        results = self.index_manager.search(query, k)
        
        # Filter by threshold
        if threshold > 0:
            results = [(id, sim) for id, sim in results if sim >= threshold]
            
        return results

    def vector_similarity(self, vec1, vec2, metric="cosine"):
        """
        Calculate the similarity between two vectors using the specified metric.
        
        Parameters:
        -----------
        vec1 : array-like
            First vector
        vec2 : array-like
            Second vector
        metric : str
            Similarity metric ('cosine', 'euclidean', etc.)
            
        Returns:
        --------
        float
            Similarity score (higher is more similar for cosine, lower for distance metrics)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if metric == "cosine":
            return 1 - distance.cosine(vec1, vec2)
        elif metric == "euclidean":
            return -distance.euclidean(vec1, vec2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

