"""Reranking module for post-processing FAISS results."""

import time
import random
from typing import List, Dict, Set
from collections import defaultdict
from .metrics import RERANK_LATENCY


class Reranker:
    """Reranks FAISS results with diversity and freshness heuristics."""
    
    def __init__(self, fusion_alpha: float = 0.7):
        self.fusion_alpha = fusion_alpha
        # Set random seed for deterministic results
        random.seed(42)
    
    def rerank(
        self,
        candidates: List[str],
        user_id: str,
        recent_items: Set[str],
        n: int = 50
    ) -> List[Dict[str, float]]:
        """Rerank candidates with diversity and freshness."""
        start_time = time.time()
        
        try:
            # Remove recent items to avoid repeats
            filtered_candidates = [item for item in candidates if item not in recent_items]
            
            # Apply diversity penalty (artist-based grouping)
            scored_items = self._apply_diversity_penalty(filtered_candidates)
            
            # Apply freshness boost (placeholder for recency/popularity)
            scored_items = self._apply_freshness_boost(scored_items)
            
            # Sort by final score
            scored_items.sort(key=lambda x: x["score"], reverse=True)
            
            # Take top N
            result = scored_items[:n]
            
            # Record latency
            latency_ms = (time.time() - start_time) * 1000
            RERANK_LATENCY.observe(latency_ms)
            
            return result
            
        except Exception as e:
            print(f"Reranking error: {e}")
            # Fallback: return first N candidates with default scores
            return [{"track_id": item, "score": 1.0 - i * 0.01} for i, item in enumerate(candidates[:n])]
    
    def _apply_diversity_penalty(self, candidates: List[str]) -> List[Dict[str, float]]:
        """Apply diversity penalty to avoid too many items from same artist."""
        # Group by artist (simplified: use first part of track_id as artist)
        artist_counts = defaultdict(int)
        scored_items = []
        
        for item in candidates:
            # Extract artist from track_id (assumes format like "artist_track")
            artist = item.split("_")[0] if "_" in item else "unknown"
            
            # Base score decreases with position
            base_score = 1.0 - (len(scored_items) * 0.01)
            
            # Apply diversity penalty
            diversity_penalty = min(0.3, artist_counts[artist] * 0.1)
            final_score = base_score - diversity_penalty
            
            scored_items.append({
                "track_id": item,
                "score": max(0.1, final_score),  # Ensure minimum score
                "artist": artist
            })
            
            artist_counts[artist] += 1
        
        return scored_items
    
    def _apply_freshness_boost(self, items: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Apply freshness boost (placeholder for recency/popularity features)."""
        # Placeholder: random boost for demonstration
        # In production, this would use actual recency/popularity features
        for item in items:
            # Small random boost to simulate freshness
            freshness_boost = random.uniform(0.0, 0.05)
            item["score"] += freshness_boost
        
        return items

