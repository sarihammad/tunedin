"""Redis cache utilities for user embeddings and recent history."""

import json
import pickle
from typing import List, Optional, Set
import redis
import numpy as np
from .metrics import update_cache_metrics


class CacheManager:
    """Manages Redis caching for user embeddings and recent history."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.embedding_ttl = 3600 * 24  # 24 hours
        self.history_ttl = 3600 * 24 * 7  # 7 days
    
    def get_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Get user embedding from cache."""
        try:
            key = f"user:vec:{user_id}"
            data = self.redis.get(key)
            if data:
                embedding = pickle.loads(data)
                update_cache_metrics(hit=True)
                return embedding
            else:
                update_cache_metrics(hit=False)
                return None
        except Exception as e:
            print(f"Cache get error for user {user_id}: {e}")
            update_cache_metrics(hit=False)
            return None
    
    def set_user_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """Cache user embedding."""
        try:
            key = f"user:vec:{user_id}"
            data = pickle.dumps(embedding)
            return self.redis.setex(key, self.embedding_ttl, data)
        except Exception as e:
            print(f"Cache set error for user {user_id}: {e}")
            return False
    
    def get_recent_items(self, user_id: str) -> Set[str]:
        """Get recent items for a user to avoid repeats."""
        try:
            key = f"user:recent:{user_id}"
            items = self.redis.smembers(key)
            return {item.decode('utf-8') for item in items}
        except Exception as e:
            print(f"Cache get recent error for user {user_id}: {e}")
            return set()
    
    def add_recent_item(self, user_id: str, track_id: str) -> bool:
        """Add item to user's recent history."""
        try:
            key = f"user:recent:{user_id}"
            # Add to set and set expiration
            pipe = self.redis.pipeline()
            pipe.sadd(key, track_id)
            pipe.expire(key, self.history_ttl)
            results = pipe.execute()
            return results[0] > 0  # True if item was added (not already present)
        except Exception as e:
            print(f"Cache add recent error for user {user_id}: {e}")
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            info = self.redis.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            print(f"Cache stats error: {e}")
            return {}

