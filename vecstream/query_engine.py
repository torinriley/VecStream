"""
QueryEngine class for high-level vector query operations.
"""

from typing import List, Tuple, Optional, Dict, Any
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

    def search(self, query: List[float], k: int = 5, threshold: float = 0.0,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors with optional filtering.
        
        Args:
            query: Query vector to find similar vectors for
            k: Number of similar vectors to return
            threshold: Minimum similarity score (0 to 1) for results
            filter_metadata: Optional metadata filters to apply
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity (highest first)
        """
        # Update index before searching
        self.index_manager.update_index()
        
        # If we don't need to filter by metadata, use standard search
        if filter_metadata is None:
            return self.index_manager.search(query, k, threshold)
        
        # Get all results above threshold for filtering
        # We need to retrieve more results than k because some might be filtered out
        get_count = min(max(k * 4, 100), len(self.index_manager.store.vectors))
        results = self.index_manager.search(query, get_count, threshold)
        
        # Filter results by metadata
        filtered_results = []
        for id, similarity in results:
            _, metadata = self.index_manager.store.get_vector_with_metadata(id)
            if metadata and self._matches_filter(metadata, filter_metadata):
                filtered_results.append((id, similarity))
                # Stop once we have k results
                if len(filtered_results) >= k:
                    break
                    
        return filtered_results[:k]

    def _matches_filter(self, metadata: Dict[str, Any], filter_query: Dict[str, Any]) -> bool:
        """Check if metadata matches the filter query.
        
        Args:
            metadata: Vector metadata
            filter_query: Filter criteria
            
        Returns:
            True if metadata matches filter, False otherwise
        """
        for key, value in filter_query.items():
            # Handle nested keys with dot notation (e.g., "user.name")
            if "." in key:
                parts = key.split(".")
                current = metadata
                match = True
                
                for part in parts[:-1]:
                    if part not in current or not isinstance(current[part], dict):
                        match = False
                        break
                    current = current[part]
                
                if not match or parts[-1] not in current:
                    return False
                
                # Check value comparison for the last part
                if current[parts[-1]] != value:
                    return False
            else:
                # Simple key-value match
                if key not in metadata or metadata[key] != value:
                    return False
        
        return True

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

