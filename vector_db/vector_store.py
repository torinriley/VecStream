import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class VectorStore:
    """Manages in-memory vector storage"""
    
    def __init__(self):
        self.vectors = {}  # id -> vector mapping
        self.metadata = {}  # id -> metadata mapping
    
    def add(self, id: str, vector: List[float], metadata: Optional[Dict] = None) -> bool:
        """Add a vector to the store"""
        if not isinstance(vector, (list, np.ndarray)):
            raise TypeError("Vector must be a list or numpy array")
            
        self.vectors[id] = np.array(vector, dtype=np.float32)
        if metadata:
            self.metadata[id] = metadata
        return True
    
    def get(self, id: str) -> Optional[np.ndarray]:
        """Retrieve a vector by id"""
        return self.vectors.get(id)
    
    def delete(self, id: str) -> bool:
        """Delete a vector from the store"""
        if id in self.vectors:
            del self.vectors[id]
            if id in self.metadata:
                del self.metadata[id]
            return True
        return False
    
    def list_ids(self) -> List[str]:
        """List all vector ids in the store"""
        return list(self.vectors.keys())
    
    def count(self) -> int:
        """Return the count of vectors in the store"""
        return len(self.vectors)
