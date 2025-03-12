from typing import List, Dict, Any, Optional, Tuple
import json
from vector_db.vector_store import VectorStore
from vector_db.index_manager import IndexManager, IndexType
from vector_db.query_engine import QueryEngine

class CommandInterface:
    """Redis-like command parser and executor"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.index_manager = IndexManager(self.vector_store)
        self.query_engine = QueryEngine(self.vector_store, self.index_manager)
        
        # Register available commands
        self.commands = {
            "VADD": self.cmd_vector_add,
            "VGET": self.cmd_vector_get,
            "VDEL": self.cmd_vector_delete,
            "VSEARCH": self.cmd_vector_search,
            "VCOUNT": self.cmd_vector_count,
            "VLIST": self.cmd_vector_list,
            "VINDEX": self.cmd_vector_build_index,
        }
    
    def execute_command(self, cmd_line: str) -> Any:
        """Parse and execute a command string"""
        parts = cmd_line.strip().split()
        if not parts:
            return "ERROR: Empty command"
            
        cmd = parts[0].upper()
        args = parts[1:]
        
        if cmd not in self.commands:
            return f"ERROR: Unknown command '{cmd}'"
            
        try:
            return self.commands[cmd](args)
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def cmd_vector_add(self, args: List[str]) -> str:
        """Add a vector: VADD key [vector_values] [METADATA json_metadata]"""
        if len(args) < 2:
            return "ERROR: VADD requires at least key and vector"
            
        key = args[0]
        
        # Parse vector values
        try:
            vector_end_idx = len(args)
            metadata = None
            
            # Check for METADATA keyword
            if "METADATA" in args:
                metadata_idx = args.index("METADATA")
                vector_end_idx = metadata_idx
                metadata_json = " ".join(args[metadata_idx+1:])
                metadata = json.loads(metadata_json)
                
            vector = [float(x) for x in args[1:vector_end_idx]]
            self.vector_store.add(key, vector, metadata)
            
            return "OK"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def cmd_vector_get(self, args: List[str]) -> str:
        """Get a vector: VGET key"""
        if len(args) != 1:
            return "ERROR: VGET requires exactly one key"
            
        key = args[0]
        vector = self.vector_store.get(key)
        
        if vector is None:
            return "(nil)"
        return str(vector.tolist())
    
    def cmd_vector_delete(self, args: List[str]) -> str:
        """Delete a vector: VDEL key"""
        if len(args) != 1:
            return "ERROR: VDEL requires exactly one key"
            
        key = args[0]
        success = self.vector_store.delete(key)
        
        return "1" if success else "0"
    
    def cmd_vector_search(self, args: List[str]) -> str:
        """Search for similar vectors: VSEARCH [vector_values] LIMIT n [METRIC metric]"""
        if len(args) < 3 or "LIMIT" not in args:
            return "ERROR: VSEARCH requires vector values and LIMIT"
            
        limit_idx = args.index("LIMIT")
        vector = [float(x) for x in args[:limit_idx]]
        limit = int(args[limit_idx+1])
        
        metric = "cosine"
        if "METRIC" in args:
            metric_idx = args.index("METRIC")
            metric = args[metric_idx+1]
            
        results = self.query_engine.search(vector, limit, metric)
        
        # Format results
        formatted = []
        for key, score in results:
            formatted.append(f"{key}: {score:.6f}")
            
        return "\n".join(formatted) if formatted else "(empty list)"
    
    def cmd_vector_count(self, args: List[str]) -> str:
        """Count vectors: VCOUNT"""
        return str(self.vector_store.count())
    
    def cmd_vector_list(self, args: List[str]) -> str:
        """List vector IDs: VLIST"""
        keys = self.vector_store.list_ids()
        return "\n".join(keys) if keys else "(empty list)"
    
    def cmd_vector_build_index(self, args: List[str]) -> str:
        """Build search index: VINDEX [TYPE index_type]"""
        index_type = IndexType.FLAT
        
        if len(args) >= 2 and args[0].upper() == "TYPE":
            try:
                index_type = IndexType(args[1].lower())
            except ValueError:
                return f"ERROR: Unknown index type '{args[1]}'"
                
        success = self.index_manager.build_index(index_type)
        
        return "OK" if success else "ERROR: Failed to build index"
