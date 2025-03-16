"""
PersistentVectorStore class for persistent vector storage.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from .vector_store import VectorStore

class PersistentVectorStore(VectorStore):
    """A class for persistent vector storage that extends VectorStore."""

    def __init__(self, filepath: str):
        """Initialize a persistent vector store.
        
        Args:
            filepath: Path to the file where vectors will be stored
        """
        super().__init__()
        self.filepath = filepath
        if os.path.exists(filepath):
            self.load()

    def add_vector(self, id: str, vector: List[float]) -> None:
        """Add a vector to the store and save to disk.
        
        Args:
            id: Unique identifier for the vector
            vector: List of float values representing the vector
            
        Raises:
            ValueError: If vector dimensions don't match existing vectors
        """
        super().add_vector(id, vector)
        self.save()

    def remove_vector(self, id: str) -> None:
        """Remove a vector from the store and update the file.
        
        Args:
            id: The vector's identifier
            
        Raises:
            KeyError: If the ID is not found
        """
        super().remove_vector(id)
        self.save()

    def save(self) -> None:
        """Save the current state to disk."""
        data = {
            'dimension': self.dimension,
            'vectors': {
                id: vector.tolist() for id, vector in self.vectors.items()
            }
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        # Save to file
        with open(self.filepath, 'w') as f:
            json.dump(data, f)

    def load(self) -> None:
        """Load the state from disk."""
        if not os.path.exists(self.filepath):
            return
            
        with open(self.filepath, 'r') as f:
            data = json.load(f)
            
        self.dimension = data.get('dimension')
        self.vectors = {
            id: np.array(vector, dtype=np.float32)
            for id, vector in data.get('vectors', {}).items()
        } 