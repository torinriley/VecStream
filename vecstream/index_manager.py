import numpy as np
from scipy.spatial import distance

class IndexManager:
    """
    A class that handles indexing strategies for efficient vector retrieval.
    """
    
    def __init__(self, vector_store):
        """
        Initialize with a reference to a VectorStore.
        
        Parameters:
        -----------
        vector_store : VectorStore
            The vector store to be indexed
        """
        self.vector_store = vector_store
        self.index = None
        self.indexed_ids = []
        self.metric = None
        
    def create_index(self, metric="cosine"):
        """
        Create a new index using the specified similarity metric.
        
        Parameters:
        -----------
        metric : str
            The similarity metric to use ('cosine', 'euclidean', etc.)
        """
        self.metric = metric
        self.indexed_ids = list(self.vector_store.vectors.keys())
        vectors = [self.vector_store.vectors[id] for id in self.indexed_ids]
        
        # Simple implementation - in production you might use libraries like FAISS, Annoy, etc.
        self.index = np.array(vectors) if vectors else np.array([])
        
    def update_index(self):
        """
        Update the index with any new vectors.
        """
        current_ids = set(self.vector_store.vectors.keys())
        indexed_ids_set = set(self.indexed_ids)
        
        # Find new vectors
        new_ids = current_ids - indexed_ids_set
        
        if new_ids:
            self.indexed_ids = list(current_ids)
            new_vectors = [self.vector_store.vectors[id] for id in new_ids]
            if self.index.size == 0:
                self.index = np.array(new_vectors)
            else:
                self.index = np.vstack((self.index, new_vectors))

