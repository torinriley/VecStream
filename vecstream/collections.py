"""
Collections module for managing multiple vector collections/namespaces.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Set
import shutil

from .binary_store import BinaryVectorStore
from .hnsw_index import HNSWIndex


class Collection:
    """A collection/namespace for organizing vectors."""
    
    def __init__(self, name: str, storage_dir: str, use_hnsw: bool = True, 
                 hnsw_params: Optional[Dict[str, Any]] = None):
        """Initialize a collection.
        
        Args:
            name: Collection name
            storage_dir: Base storage directory
            use_hnsw: Whether to use HNSW indexing (default: True)
            hnsw_params: Optional parameters for HNSW index
        """
        self.name = name
        self.collection_dir = os.path.join(storage_dir, "collections", name)
        os.makedirs(self.collection_dir, exist_ok=True)
        
        # Create the store
        self.store = BinaryVectorStore(self.collection_dir)
        
        # HNSW index settings
        self.use_hnsw = use_hnsw
        self.hnsw_params = hnsw_params or {}
        self.hnsw_index = None
        
        # Initialize HNSW index if required
        if use_hnsw and self.store.dimension is not None:
            self._init_hnsw_index()
    
    def _init_hnsw_index(self) -> None:
        """Initialize the HNSW index with current vectors."""
        if self.store.dimension is None:
            return
            
        # Create HNSW index with parameters
        M = self.hnsw_params.get('M', 16)
        ef_construction = self.hnsw_params.get('ef_construction', 200)
        ml = self.hnsw_params.get('ml', None)
        
        self.hnsw_index = HNSWIndex(
            dim=self.store.dimension,
            M=M,
            ef_construction=ef_construction,
            ml=ml
        )
        
        # Add all existing vectors to the index
        for id, vector in self.store.vectors.items():
            self.hnsw_index.add_item(id, vector)
    
    def add_vector(self, id: str, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a vector to the collection.
        
        Args:
            id: Unique identifier for the vector
            vector: The vector to add
            metadata: Optional metadata for the vector
        """
        # Add to binary store
        self.store.add_vector(id, vector, metadata)
        
        # Add to HNSW index if enabled
        if self.use_hnsw:
            if self.hnsw_index is None:
                self._init_hnsw_index()
            else:
                self.hnsw_index.add_item(id, vector)
    
    def remove_vector(self, id: str) -> None:
        """Remove a vector from the collection.
        
        Args:
            id: The vector's identifier
        """
        # Remove from store
        self.store.remove_vector(id)
        
        # Remove from HNSW index if enabled
        if self.use_hnsw and self.hnsw_index is not None:
            try:
                self.hnsw_index.remove_item(id)
            except KeyError:
                pass  # Already removed or not in index
    
    def search_similar(self, query: List[float], k: int = 5, threshold: float = 0.0,
                        ef_search: Optional[int] = None, 
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Search for similar vectors with optional filtering.
        
        Args:
            query: Query vector
            k: Number of results to return
            threshold: Minimum similarity score threshold
            ef_search: HNSW ef search parameter (only used with HNSW)
            filter_metadata: Optional metadata filtering criteria
            
        Returns:
            List of (id, similarity) tuples
        """
        # If using HNSW index
        if self.use_hnsw and self.hnsw_index is not None:
            results = self.hnsw_index.search(query, k=k if filter_metadata is None else max(k * 10, 100), 
                                            ef_search=ef_search)
            
            # Apply metadata filtering if needed
            if filter_metadata:
                filtered_results = []
                for id, similarity in results:
                    if similarity >= threshold:
                        _, metadata = self.store.get_vector_with_metadata(id)
                        if metadata and self._matches_filter(metadata, filter_metadata):
                            filtered_results.append((id, similarity))
                            if len(filtered_results) >= k:
                                break
                return filtered_results[:k]
            else:
                # Just apply threshold and return
                return [(id, sim) for id, sim in results if sim >= threshold][:k]
        else:
            # Use regular search from store
            if filter_metadata:
                # Need to manually filter results
                all_results = self.store.search_similar(query, k=len(self.store.vectors), threshold=threshold)
                filtered_results = []
                
                for id, similarity in all_results:
                    _, metadata = self.store.get_vector_with_metadata(id)
                    if metadata and self._matches_filter(metadata, filter_metadata):
                        filtered_results.append((id, similarity))
                        if len(filtered_results) >= k:
                            break
                
                return filtered_results[:k]
            else:
                # No filtering needed
                return self.store.search_similar(query, k=k, threshold=threshold)
    
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
    
    def get_vector_with_metadata(self, id: str) -> Tuple[List[float], Optional[Dict[str, Any]]]:
        """Get a vector and its metadata.
        
        Args:
            id: Vector identifier
            
        Returns:
            Tuple of (vector, metadata)
        """
        return self.store.get_vector_with_metadata(id)
    
    def get_vector_count(self) -> int:
        """Get the number of vectors in the collection."""
        return len(self.store.vectors)


class CollectionManager:
    """Manager for multiple vector collections."""
    
    def __init__(self, base_storage_dir: str, use_hnsw: bool = True, 
                 default_hnsw_params: Optional[Dict[str, Any]] = None):
        """Initialize the collection manager.
        
        Args:
            base_storage_dir: Base directory for storing collections
            use_hnsw: Whether to use HNSW indexing by default
            default_hnsw_params: Default parameters for HNSW index
        """
        self.base_storage_dir = base_storage_dir
        self.collections_dir = os.path.join(base_storage_dir, "collections")
        self.metadata_file = os.path.join(base_storage_dir, "collections_metadata.json")
        
        # Create directories
        os.makedirs(self.collections_dir, exist_ok=True)
        
        # Default HNSW settings
        self.use_hnsw = use_hnsw
        self.default_hnsw_params = default_hnsw_params or {
            "M": 16,
            "ef_construction": 200
        }
        
        # Active collections
        self.collections: Dict[str, Collection] = {}
        
        # Collection metadata
        self.collections_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing collections metadata
        self._load_metadata()
        
    def _load_metadata(self) -> None:
        """Load collections metadata from file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    self.collections_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load collections metadata: {str(e)}")
                self.collections_metadata = {}
    
    def _save_metadata(self) -> None:
        """Save collections metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.collections_metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save collections metadata: {str(e)}")
    
    def create_collection(self, name: str, use_hnsw: Optional[bool] = None, 
                         hnsw_params: Optional[Dict[str, Any]] = None) -> Collection:
        """Create a new collection.
        
        Args:
            name: Collection name
            use_hnsw: Whether to use HNSW indexing (defaults to manager setting)
            hnsw_params: Custom HNSW parameters (defaults to manager settings)
            
        Returns:
            The created collection
            
        Raises:
            ValueError: If collection already exists
        """
        if name in self.collections or name in self.collections_metadata:
            raise ValueError(f"Collection '{name}' already exists")
        
        # Use defaults if not specified
        use_hnsw_flag = self.use_hnsw if use_hnsw is None else use_hnsw
        hnsw_params_dict = dict(self.default_hnsw_params) if hnsw_params is None else hnsw_params
        
        # Create and store collection
        collection = Collection(name, self.base_storage_dir, use_hnsw_flag, hnsw_params_dict)
        self.collections[name] = collection
        
        # Save metadata
        self.collections_metadata[name] = {
            "name": name,
            "created_at": __import__('datetime').datetime.now().isoformat(),
            "use_hnsw": use_hnsw_flag,
            "hnsw_params": hnsw_params_dict
        }
        self._save_metadata()
        
        return collection
    
    def get_collection(self, name: str) -> Collection:
        """Get a collection by name.
        
        Args:
            name: Collection name
            
        Returns:
            The collection
            
        Raises:
            KeyError: If collection doesn't exist
        """
        # Return if already loaded
        if name in self.collections:
            return self.collections[name]
        
        # Check if collection exists in metadata
        if name not in self.collections_metadata:
            raise KeyError(f"Collection '{name}' does not exist")
        
        # Load collection metadata
        metadata = self.collections_metadata[name]
        use_hnsw = metadata.get("use_hnsw", self.use_hnsw)
        hnsw_params = metadata.get("hnsw_params", self.default_hnsw_params)
        
        # Create and return collection
        collection = Collection(name, self.base_storage_dir, use_hnsw, hnsw_params)
        self.collections[name] = collection
        
        return collection
    
    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        return list(self.collections_metadata.keys())
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection.
        
        Args:
            name: Collection name
            
        Raises:
            KeyError: If collection doesn't exist
        """
        if name not in self.collections_metadata:
            raise KeyError(f"Collection '{name}' does not exist")
        
        # Remove from active collections
        if name in self.collections:
            del self.collections[name]
        
        # Remove from metadata
        del self.collections_metadata[name]
        self._save_metadata()
        
        # Delete collection directory
        collection_dir = os.path.join(self.collections_dir, name)
        if os.path.exists(collection_dir):
            shutil.rmtree(collection_dir)
    
    def get_collection_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a collection.
        
        Args:
            name: Collection name
            
        Returns:
            Dictionary with collection statistics
            
        Raises:
            KeyError: If collection doesn't exist
        """
        collection = self.get_collection(name)
        vectors_size, metadata_size = collection.store.get_store_size()
        
        stats = {
            "name": name,
            "vector_count": collection.get_vector_count(),
            "vectors_size_bytes": vectors_size,
            "metadata_size_bytes": metadata_size,
            "total_size_bytes": vectors_size + metadata_size,
            "using_hnsw": collection.use_hnsw,
            "dimension": collection.store.dimension,
            **self.collections_metadata.get(name, {})
        }
        
        return stats 