"""
HNSW (Hierarchical Navigable Small World) index implementation for efficient approximate nearest neighbor search.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import heapq
import random

class HNSWIndex:
    """HNSW index for efficient approximate nearest neighbor search."""
    
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, ml: int = None):
        """Initialize HNSW index.
        
        Args:
            dim: Dimensionality of vectors
            M: Maximum number of connections per node (default: 16)
            ef_construction: Size of dynamic candidate list for index construction (default: 200)
            ml: Maximum level for the graph (if None, computed automatically)
        """
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # Max connections for layer 0 (typically 2*M)
        self.ef_construction = ef_construction
        self.ml = ml if ml is not None else int(np.log2(1000))  # Max level, calculated based on expected DB size
        
        # Graph data structures
        self.nodes: Dict[str, np.ndarray] = {}  # ID -> vector
        self.node_levels: Dict[str, int] = {}  # ID -> level
        self.graphs: Dict[int, Dict[str, Set[str]]] = {}  # level -> (id -> set of neighbor IDs)
        
        # Initialize empty graphs for each level
        for level in range(self.ml + 1):
            self.graphs[level] = {}
        
        # Entry point
        self.ep = None  # ID of entry point
    
    def _get_random_level(self) -> int:
        """Assign a random level to a new element using exponential distribution."""
        return int(-np.log(random.random()) * self.M / self.M_max0)
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate distance between vectors (1 - cosine similarity)."""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm == 0 or b_norm == 0:
            return 1.0
        
        cos_sim = np.dot(a, b) / (a_norm * b_norm)
        # Convert similarity to distance (1 - similarity), clamping to [0, 2]
        return max(0.0, min(2.0, 1.0 - cos_sim))
    
    def _search_layer(self, q: np.ndarray, ep: str, ef: int, level: int) -> List[Tuple[float, str]]:
        """Search for nearest neighbors in a single layer.
        
        Args:
            q: Query vector
            ep: Entry point ID
            ef: Size of dynamic candidate list
            level: Level to search
            
        Returns:
            List of (distance, id) tuples for nearest neighbors
        """
        # Set of visited elements
        visited = set([ep])
        
        # Priority queue for the candidate set, sorted by distance
        # Use a min-heap for candidates we're expanding (smaller distances first)
        candidates = [(self._distance(q, self.nodes[ep]), ep)]
        heapq.heapify(candidates)
        
        # Use a max-heap for results (larger distances are popped first when exceeding ef)
        results = [(-self._distance(q, self.nodes[ep]), ep)]
        heapq.heapify(results)
        
        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            
            # Get the farthest result distance (negative because we use max-heap)
            furthest_dist = -results[0][0] if results else float('inf')
            
            # If candidate is farther than our worst result, we're done
            if c_dist > furthest_dist and len(results) >= ef:
                break
            
            # Process neighbors of current candidate
            for neighbor_id in self.graphs[level].get(c_id, set()):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    d = self._distance(q, self.nodes[neighbor_id])
                    
                    # If we haven't filled our results yet, or this is closer than the worst result
                    if len(results) < ef or d < furthest_dist:
                        heapq.heappush(candidates, (d, neighbor_id))
                        heapq.heappush(results, (-d, neighbor_id))
                        
                        # If we've exceeded ef, remove the worst element
                        if len(results) > ef:
                            heapq.heappop(results)
        
        # Convert results to (distance, id) format, sorted by distance
        return [(self._distance(q, self.nodes[id]), id) for _, id in sorted([(d, id) for d, id in results])]
    
    def _select_neighbors(self, q: np.ndarray, candidates: List[Tuple[float, str]], M: int) -> List[str]:
        """Select the M closest neighbors from candidates.
        
        Args:
            q: Query vector
            candidates: List of (distance, id) tuples
            M: Maximum number of neighbors to return
            
        Returns:
            List of selected neighbor IDs
        """
        # Simple heuristic: just select M closest neighbors
        candidates.sort()  # Sort by distance
        return [id for _, id in candidates[:M]]
    
    def add_item(self, id: str, vector: np.ndarray) -> None:
        """Add a new item to the index.
        
        Args:
            id: Unique identifier for the vector
            vector: Vector to add
        """
        if id in self.nodes:
            # Update the vector if the ID already exists
            self.nodes[id] = vector
            return
        
        self.nodes[id] = vector
        
        # Get random level for this node using exponential distribution
        l = min(self._get_random_level(), self.ml)
        self.node_levels[id] = l
        
        # If this is the first node, make it the entry point and return
        if self.ep is None:
            self.ep = id
            # Initialize empty neighbor sets for this node at all levels up to l
            for level in range(l + 1):
                self.graphs[level][id] = set()
            return
        
        # Find insertion point and add connections
        ep = self.ep
        ep_level = self.node_levels[self.ep]
        
        # For levels above l, just search without adding connections
        for level in range(min(ep_level, self.ml), l, -1):
            res = self._search_layer(vector, ep, 1, level)
            if res:
                ep = res[0][1]  # ID of the closest element
        
        # For levels where we add the node
        for level in range(min(l, ep_level), -1, -1):
            # Find ef_construction nearest elements at this level
            W = self._search_layer(vector, ep, self.ef_construction, level)
            # Select M neighbors
            neighbors = self._select_neighbors(vector, W, self.M_max0 if level == 0 else self.M)
            
            # Initialize an empty set for this node's neighbors at this level
            if id not in self.graphs[level]:
                self.graphs[level][id] = set()
            
            # Add bidirectional connections
            for neighbor_id in neighbors:
                self.graphs[level][id].add(neighbor_id)
                
                # Create neighbor set for neighbor if it doesn't exist
                if neighbor_id not in self.graphs[level]:
                    self.graphs[level][neighbor_id] = set()
                
                self.graphs[level][neighbor_id].add(id)
                
                # Ensure neighbor doesn't have too many connections
                if len(self.graphs[level][neighbor_id]) > (self.M_max0 if level == 0 else self.M):
                    # Need to remove some connections
                    # For simplicity, just keep the M closest ones
                    neighbor_candidates = [(self._distance(self.nodes[neighbor_id], self.nodes[n_id]), n_id) 
                                          for n_id in self.graphs[level][neighbor_id]]
                    keep_neighbors = self._select_neighbors(self.nodes[neighbor_id], neighbor_candidates, 
                                                           self.M_max0 if level == 0 else self.M)
                    self.graphs[level][neighbor_id] = set(keep_neighbors)
            
            # Update entry point for the next level
            ep = id
        
        # Update entry point if this node has higher level
        if l > ep_level:
            self.ep = id
    
    def remove_item(self, id: str) -> None:
        """Remove an item from the index.
        
        Args:
            id: Unique identifier for the vector to remove
            
        Raises:
            KeyError: If the ID doesn't exist
        """
        if id not in self.nodes:
            raise KeyError(f"Item with ID {id} not found in index")
        
        l = self.node_levels[id]
        
        # Remove connections at all levels
        for level in range(l + 1):
            # Remove outgoing connections
            neighbors = self.graphs[level].get(id, set())
            for neighbor_id in neighbors:
                if neighbor_id in self.graphs[level]:
                    self.graphs[level][neighbor_id].discard(id)
            
            # Remove the node from the graph at this level
            if id in self.graphs[level]:
                del self.graphs[level][id]
        
        # Remove from nodes and levels
        del self.nodes[id]
        del self.node_levels[id]
        
        # Update entry point if needed
        if self.ep == id:
            if not self.nodes:  # If this was the last node
                self.ep = None
            else:
                # Find a new entry point with the highest level
                max_level = -1
                new_ep = None
                for node_id, level in self.node_levels.items():
                    if level > max_level:
                        max_level = level
                        new_ep = node_id
                self.ep = new_ep
    
    def search(self, query: np.ndarray, k: int = 10, ef_search: int = None) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of nearest neighbors to return
            ef_search: Size of dynamic candidate list for search (default: max(ef_construction, k))
            
        Returns:
            List of (id, similarity) tuples sorted by similarity (highest first)
        """
        if not self.nodes or self.ep is None:
            return []
        
        ef = max(k, self.ef_construction if ef_search is None else ef_search)
        
        # Start from the top layer of the entry point
        ep = self.ep
        L = self.node_levels[ep]
        
        # Traverse from top to bottom, finding entry point for next level each time
        for level in range(L, 0, -1):
            res = self._search_layer(query, ep, 1, level)
            if res:
                ep = res[0][1]  # ID of the closest element
        
        # Search at layer 0 with full ef value
        candidates = self._search_layer(query, ep, ef, 0)
        
        # Process results - convert distances to similarities and cap at k results
        results = []
        for distance, id in candidates[:k]:
            # Convert distance back to similarity (1 - distance)
            similarity = 1.0 - distance
            results.append((id, similarity))
        
        return results 