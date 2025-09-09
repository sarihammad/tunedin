"""FAISS index management for efficient similarity search."""

import os
import numpy as np
import faiss
from typing import List, Tuple, Optional
from .metrics import FAISS_LATENCY
import time


class FAISSIndex:
    """Manages FAISS index for item similarity search."""
    
    def __init__(self, index_path: str, item_ids_path: str):
        self.index_path = index_path
        self.item_ids_path = item_ids_path
        self.index: Optional[faiss.Index] = None
        self.item_ids: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
    
    def load(self) -> bool:
        """Load FAISS index and item IDs from disk."""
        try:
            # Load FAISS index
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                self.dimension = self.index.d
                print(f"Loaded FAISS index with {self.index.ntotal} items, dim={self.dimension}")
            else:
                print(f"FAISS index not found at {self.index_path}")
                return False
            
            # Load item IDs
            if os.path.exists(self.item_ids_path):
                self.item_ids = np.load(self.item_ids_path)
                print(f"Loaded {len(self.item_ids)} item IDs")
            else:
                print(f"Item IDs not found at {self.item_ids_path}")
                return False
            
            # Validate dimensions match
            if len(self.item_ids) != self.index.ntotal:
                print(f"Dimension mismatch: {len(self.item_ids)} item IDs vs {self.index.ntotal} index items")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return False
    
    def search(self, query_vector: np.ndarray, k: int = 200) -> List[str]:
        """Search for similar items using FAISS."""
        if self.index is None or self.item_ids is None:
            raise RuntimeError("FAISS index not loaded")
        
        if query_vector.ndim != 2 or query_vector.shape[1] != self.dimension:
            raise ValueError(f"Query vector must be 2D with dimension {self.dimension}")
        
        start_time = time.time()
        
        try:
            # FAISS search
            scores, indices = self.index.search(query_vector.astype(np.float32), k)
            
            # Convert indices to item IDs
            item_ids = [str(self.item_ids[idx]) for idx in indices[0] if idx != -1]
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            FAISS_LATENCY.observe(latency_ms)
            
            return item_ids
            
        except Exception as e:
            print(f"FAISS search error: {e}")
            raise
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        if self.index is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "total_items": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": type(self.index).__name__,
        }

