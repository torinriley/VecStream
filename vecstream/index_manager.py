"""
IndexManager class for managing vector indices for efficient similarity search.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from .vector_store import VectorStore
from .hnsw_index import HNSWIndex

class IndexManager:
    """A class for managing vector indices for efficient similarity search."""

    def __init__(self, store: VectorStore, use_hnsw: bool = True, 
                 hnsw_params: Optional[Dict[str, Any]] = None):
        """Initialize the index manager.
        
        Args:
            store: VectorStore instance to manage indices for
            use_hnsw: Whether to use HNSW indexing (default: True)
            hnsw_params: Optional parameters for HNSW index
        """
        self.store = store
        self.use_hnsw = use_hnsw
        self.hnsw_params = hnsw_params or {"M": 16, "ef_construction": 200}
        
        # Initialize indices
        self.index = None  # Standard index (normalized vectors)
        self.hnsw_index = None  # HNSW index
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

        # Convert to numpy array and normalize for standard index
        vectors_array = np.array(vectors)
        norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.index = vectors_array / norms
        
        # Create HNSW index if enabled
        if self.use_hnsw and self.store.dimension is not None:
            self._create_hnsw_index()

    def _create_hnsw_index(self) -> None:
        """Create HNSW index with current vectors."""
        if self.store.dimension is None:
            return
        
        # Create new HNSW index with parameters
        M = self.hnsw_params.get("M", 16)
        ef_construction = self.hnsw_params.get("ef_construction", 200)
        ml = self.hnsw_params.get("ml", None)
        
        self.hnsw_index = HNSWIndex(
            dim=self.store.dimension,
            M=M,
            ef_construction=ef_construction,
            ml=ml
        )
        
        # Add all vectors to the index
        for id, vector in self.store.vectors.items():
            # Convert to numpy array if needed
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
            self.hnsw_index.add_item(id, vector)

    def update_index(self) -> None:
        """Update the index with new vectors."""
        if self.index is None:
            self.create_index()
            return

        # Check for new vectors
        current_ids = set(self.store.vectors.keys())
        indexed_ids = set(self.indexed_ids)
        
        if current_ids != indexed_ids:
            # HNSW index can be updated incrementally, but for simplicity
            # we'll recreate both indices for consistency when the vector set changes
            self.create_index()

    def search(self, query: List[float], k: int = 5, threshold: float = 0.0,
               ef_search: Optional[int] = None) -> List[Tuple[str, float]]:
        """Search the index for similar vectors.
        
        Args:
            query: Query vector to find similar vectors for
            k: Number of similar vectors to return
            threshold: Minimum similarity score (0 to 1) for results
            ef_search: HNSW search parameter (only used with HNSW)
            
        Returns:
            List of (id, similarity) tuples, sorted by similarity (highest first)
        """
        if (self.index is None and self.hnsw_index is None) or not self.indexed_ids:
            return []
        
        # Use HNSW index if available for better performance
        if self.use_hnsw and self.hnsw_index is not None:
            # Convert query to numpy array
            query_array = np.array(query, dtype=np.float32)
            
            # Use HNSW index for search
            results = self.hnsw_index.search(query_array, k=k, ef_search=ef_search)
            
            # Filter by threshold if needed
            if threshold > 0:
                results = [(id, sim) for id, sim in results if sim >= threshold]
                
            return results
        else:
            # Fall back to standard search
            return self._search_standard(query, k, threshold)

    def _search_standard(self, query: List[float], k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search using standard normalized dot product.
        
        Args:
            query: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (id, similarity) tuples
        """
        # Normalize query vector
        query_array = np.array(query, dtype=np.float32)
        query_norm = np.linalg.norm(query_array)
        if query_norm > 0:
            query_array = query_array / query_norm

        # Compute similarities
        similarities = np.dot(self.index, query_array)
        
        # Filter by threshold and get top k results
        results = []
        for idx, sim in enumerate(similarities):
            if sim >= threshold:
                id = self.indexed_ids[idx]
                results.append((id, float(sim)))
        
        # Sort by similarity (highest first) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

