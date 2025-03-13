import numpy as np
from scipy.spatial import distance

class QueryEngine:
    """
    A class that performs vector similarity searches.
    """
    
    def __init__(self, index_manager):
        """
        Initialize with a reference to an IndexManager.
        
        Parameters:
        -----------
        index_manager : IndexManager
            The index manager to use for searches
        """
        self.index_manager = index_manager
        
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
            
    def search(self, query_vector, k=10, metric="cosine"):
        """
        Search for the k most similar vectors to the query vector.
        
        Parameters:
        -----------
        query_vector : array-like
            The query vector to compare against
        k : int
            Number of results to return
        metric : str
            Similarity metric to use
            
        Returns:
        --------
        list of tuples
            (id, similarity_score) pairs for the k most similar vectors
        """
        # Ensure index is up-to-date
        self.index_manager.update_index()
        
        query_vector = np.array(query_vector)
        results = []
        
        # Calculate similarities
        for i, id in enumerate(self.index_manager.indexed_ids):
            vec = self.index_manager.vector_store.get(id)
            similarity = self.vector_similarity(query_vector, vec, metric)
            results.append((id, similarity))
            
        # Sort by similarity (higher is better for cosine similarity)
        if metric == "cosine":
            results.sort(key=lambda x: x[1], reverse=True)
        else:
            # For distance metrics, lower is better
            results.sort(key=lambda x: x[1])
            
        return results[:k]

