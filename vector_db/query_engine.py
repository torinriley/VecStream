import numpy as np
from typing import Dict, List, Tuple, Optional
from vector_db.vector_store import VectorStore
from vector_db.index_manager import IndexManager

class QueryEngine:
    """Processes and executes search queries"""
    
    def __init__(self, vector_store: VectorStore, index_manager: IndexManager):
        self.vector_store = vector_store
        self.index_manager = index_manager
    
    def vector_similarity(self, vec1: np.ndarray, vec2: np.ndarray, metric: str = "cosine") -> float:
        """Calculate similarity between two vectors"""
        if metric == "cosine":
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            return np.dot(vec1, vec2) / (norm1 * norm2)
        elif metric == "euclidean":
            return 1.0 / (1.0 + np.linalg.norm(vec1 - vec2))
        elif metric == "dot":
            return np.dot(vec1, vec2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def search(self, query_vector: List[float], k: int = 5, metric: str = "cosine") -> List[Tuple[str, float]]:
        """Search for k nearest vectors to the query vector"""
        query_vector = np.array(query_vector, dtype=np.float32)
        
        # If index is not built, use the vector store directly
        if not self.index_manager.is_indexed:
            self.index_manager.build_index()
        
        # For flat index, calculate similarity with all vectors
        results = []
        for id, vector in self.vector_store.vectors.items():
            similarity = self.vector_similarity(query_vector, vector, metric)
            results.append((id, similarity))
        
        # Sort by similarity (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return results[:k]
