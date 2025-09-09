"""Dependency injection for FastAPI application."""

import os
import time
from typing import Optional
import numpy as np
import redis
from dotenv import load_dotenv
from .cache import CacheManager
from .faiss_index import FAISSIndex


# Load environment variables
load_dotenv()


class Dependencies:
    """Container for application dependencies."""
    
    def __init__(self):
        self.start_time = time.time()
        self.redis_client: Optional[redis.Redis] = None
        self.cache_manager: Optional[CacheManager] = None
        self.faiss_index: Optional[FAISSIndex] = None
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.item_ids: Optional[np.ndarray] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize all dependencies."""
        try:
            # Initialize Redis
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            
            # Test Redis connection
            self.redis_client.ping()
            print("Redis connection established")
            
            # Initialize cache manager
            self.cache_manager = CacheManager(self.redis_client)
            
            # Initialize FAISS index
            index_path = os.getenv("FAISS_INDEX_PATH", "/models/index.faiss")
            item_ids_path = os.getenv("ITEM_IDS_PATH", "/models/item_ids.npy")
            self.faiss_index = FAISSIndex(index_path, item_ids_path)
            
            # Load FAISS index
            if not self.faiss_index.load():
                print("Warning: FAISS index not loaded - recommendations will return 503")
            
            # Load embeddings
            self._load_embeddings()
            
            self._initialized = True
            print("All dependencies initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error initializing dependencies: {e}")
            return False
    
    def _load_embeddings(self) -> None:
        """Load user and item embeddings from disk."""
        try:
            # Load user embeddings
            user_emb_path = os.getenv("USER_EMB_PATH", "/models/user_emb.npy")
            if os.path.exists(user_emb_path):
                self.user_embeddings = np.load(user_emb_path)
                print(f"Loaded user embeddings: {self.user_embeddings.shape}")
            else:
                print(f"User embeddings not found at {user_emb_path}")
            
            # Load item embeddings
            item_emb_path = os.getenv("ITEM_EMB_PATH", "/models/item_emb.npy")
            if os.path.exists(item_emb_path):
                self.item_embeddings = np.load(item_emb_path)
                print(f"Loaded item embeddings: {self.item_embeddings.shape}")
            else:
                print(f"Item embeddings not found at {item_emb_path}")
            
            # Load item IDs
            item_ids_path = os.getenv("ITEM_IDS_PATH", "/models/item_ids.npy")
            if os.path.exists(item_ids_path):
                self.item_ids = np.load(item_ids_path)
                print(f"Loaded item IDs: {len(self.item_ids)}")
            else:
                print(f"Item IDs not found at {item_ids_path}")
                
        except Exception as e:
            print(f"Error loading embeddings: {e}")
    
    def get_user_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Get user vector from cache or disk."""
        if not self._initialized:
            return None
        
        # Try cache first
        if self.cache_manager:
            cached_vector = self.cache_manager.get_user_embedding(user_id)
            if cached_vector is not None:
                return cached_vector
        
        # Fallback to disk
        if self.user_embeddings is not None:
            try:
                user_idx = int(user_id)
                if 0 <= user_idx < len(self.user_embeddings):
                    vector = self.user_embeddings[user_idx]
                    # Cache for future use
                    if self.cache_manager:
                        self.cache_manager.set_user_embedding(user_id, vector)
                    return vector
            except (ValueError, IndexError):
                pass
        
        # Cold start: return None (will be handled by caller)
        return None
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time
    
    def is_healthy(self) -> bool:
        """Check if all critical dependencies are healthy."""
        if not self._initialized:
            return False
        
        # Check Redis
        try:
            if self.redis_client:
                self.redis_client.ping()
            else:
                return False
        except:
            return False
        
        # Check FAISS index
        if not self.faiss_index or not self.faiss_index.index:
            return False
        
        return True


# Global dependencies instance
deps = Dependencies()

