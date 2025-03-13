import numpy as np

class VectorStore:
    """
    A class for storing and retrieving vectors.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.vectors = {}
        
    def add(self, id, vector):
        """
        Add a vector to the store with the specified ID.
        
        Parameters:
        -----------
        id : str or int
            The unique identifier for the vector
        vector : array-like
            The vector to store
        """
        self.vectors[id] = np.array(vector)
        
    def get(self, id):
        """
        Retrieve a vector by its ID.
        
        Parameters:
        -----------
        id : str or int
            The unique identifier for the vector
            
        Returns:
        --------
        numpy.ndarray or None
            The stored vector or None if not found
        """
        return self.vectors.get(id, None)
    
    def remove(self, id):
        """
        Remove a vector from the store.
        
        Parameters:
        -----------
        id : str or int
            The unique identifier for the vector to remove
            
        Returns:
        --------
        bool
            True if removal was successful, False if ID not found
        """
        if id in self.vectors:
            del self.vectors[id]
            return True
        return False

