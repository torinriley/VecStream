"""
Binary persistence layer for VectorStore using NumPy's efficient binary format.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional
from .vector_store import VectorStore

class BinaryVectorStore(VectorStore):
    """Vector store with binary persistence using NumPy's .npy format."""
    
    def __init__(self, storage_dir: str):
        """Initialize the binary vector store.
        
        Args:
            storage_dir: Directory to store the binary files and metadata
        """
        super().__init__()
        self.storage_dir = storage_dir
        self.metadata_file = os.path.join(storage_dir, "metadata.json")
        self.vectors_file = os.path.join(storage_dir, "vectors.npy")
        self.metadata: Dict[str, dict] = {}
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_store()
    
    def _load_store(self) -> None:
        """Load vectors and metadata from disk."""
        try:
            # Load metadata
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load vectors
            if os.path.exists(self.vectors_file):
                loaded_vectors = np.load(self.vectors_file, allow_pickle=True).item()
                self.vectors = loaded_vectors
                if self.vectors:
                    # Set dimension based on first vector
                    first_vec = next(iter(self.vectors.values()))
                    self.dimension = len(first_vec)
        except Exception as e:
            print(f"Warning: Failed to load store: {str(e)}")
            self.vectors = {}
            self.metadata = {}
    
    def _save_store(self) -> None:
        """Save vectors and metadata to disk."""
        try:
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
            
            # Save vectors in binary format
            np.save(self.vectors_file, self.vectors)
        except Exception as e:
            print(f"Warning: Failed to save store: {str(e)}")
    
    def add_vector(self, id: str, vector: List[float], metadata: Optional[dict] = None) -> None:
        """Add a vector with optional metadata.
        
        Args:
            id: Unique identifier for the vector
            vector: List of float values representing the vector
            metadata: Optional dictionary of metadata associated with the vector
        """
        # Add vector to in-memory store
        super().add_vector(id, vector)
        
        # Store metadata if provided
        if metadata:
            self.metadata[id] = metadata
        
        # Save to disk
        self._save_store()
    
    def remove_vector(self, id: str) -> None:
        """Remove a vector and its metadata.
        
        Args:
            id: The vector's identifier
        """
        # Remove from in-memory store
        super().remove_vector(id)
        
        # Remove metadata if exists
        self.metadata.pop(id, None)
        
        # Save changes to disk
        self._save_store()
    
    def get_vector_with_metadata(self, id: str) -> Tuple[List[float], Optional[dict]]:
        """Get a vector and its metadata.
        
        Args:
            id: The vector's identifier
            
        Returns:
            Tuple of (vector, metadata)
        """
        vector = self.get_vector(id)
        metadata = self.metadata.get(id)
        return vector, metadata
    
    def clear_store(self) -> None:
        """Clear all vectors and metadata from store."""
        self.vectors = {}
        self.metadata = {}
        self.dimension = None
        
        # Remove files if they exist
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        if os.path.exists(self.vectors_file):
            os.remove(self.vectors_file)
    
    def get_store_size(self) -> Tuple[int, int]:
        """Get the size of the store in bytes.
        
        Returns:
            Tuple of (vectors_size, metadata_size) in bytes
        """
        vectors_size = os.path.getsize(self.vectors_file) if os.path.exists(self.vectors_file) else 0
        metadata_size = os.path.getsize(self.metadata_file) if os.path.exists(self.metadata_file) else 0
        return vectors_size, metadata_size 