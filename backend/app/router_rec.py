"""Recommendation router for user recommendations."""

from typing import List
from fastapi import APIRouter, HTTPException, Query, Depends
import numpy as np
from .deps import deps
from .schemas import RecommendationResponse, RecommendationItem
from .rerank import Reranker
from .metrics import update_cache_metrics

router = APIRouter(prefix="/rec", tags=["recommendations"])

# Initialize reranker
reranker = Reranker(fusion_alpha=0.7)


@router.get("/users/{user_id}", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: str,
    n: int = Query(50, ge=1, le=100, description="Number of recommendations"),
    deps_instance=Depends(lambda: deps)
):
    """Get personalized recommendations for a user."""
    
    # Check if dependencies are healthy
    if not deps_instance.is_healthy():
        raise HTTPException(status_code=503, detail="Service not ready - models not loaded")
    
    try:
        # Get user vector
        user_vector = deps_instance.get_user_vector(user_id)
        if user_vector is None:
            # Cold start: return popular items or random recommendations
            return await _get_cold_start_recommendations(user_id, n, deps_instance)
        
        # Ensure user_vector is 2D for FAISS
        if user_vector.ndim == 1:
            user_vector = user_vector.reshape(1, -1)
        
        # Search FAISS index
        candidates = deps_instance.faiss_index.search(user_vector, k=200)
        
        # Get recent items for deduplication
        recent_items = set()
        if deps_instance.cache_manager:
            recent_items = deps_instance.cache_manager.get_recent_items(user_id)
        
        # Rerank candidates
        reranked_items = reranker.rerank(candidates, user_id, recent_items, n)
        
        # Convert to response format
        items = [
            RecommendationItem(track_id=item["track_id"], score=item["score"])
            for item in reranked_items
        ]
        
        # Check if user vector was cached
        cache_hit = deps_instance.cache_manager is not None and \
                   deps_instance.cache_manager.get_user_embedding(user_id) is not None
        
        return RecommendationResponse(
            user_id=user_id,
            items=items,
            total=len(items),
            cache_hit=cache_hit
        )
        
    except Exception as e:
        print(f"Error getting recommendations for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


async def _get_cold_start_recommendations(
    user_id: str, 
    n: int, 
    deps_instance
) -> RecommendationResponse:
    """Get cold start recommendations for new users."""
    
    # Simple cold start: return first N items with decreasing scores
    items = []
    if deps_instance.item_ids is not None:
        for i in range(min(n, len(deps_instance.item_ids))):
            track_id = str(deps_instance.item_ids[i])
            score = 1.0 - (i * 0.01)  # Decreasing scores
            items.append(RecommendationItem(track_id=track_id, score=score))
    
    return RecommendationResponse(
        user_id=user_id,
        items=items,
        total=len(items),
        cache_hit=False
    )


@router.get("/users/{user_id}/similar", response_model=RecommendationResponse)
async def get_similar_users(
    user_id: str,
    n: int = Query(20, ge=1, le=50, description="Number of similar users"),
    deps_instance=Depends(lambda: deps)
):
    """Get users similar to the given user (placeholder implementation)."""
    
    # This is a placeholder for user-to-user similarity
    # In a full implementation, you'd compute user-user similarities
    
    items = []
    for i in range(min(n, 20)):  # Mock similar users
        similar_user_id = f"user_{i}"
        score = 1.0 - (i * 0.05)
        items.append(RecommendationItem(track_id=similar_user_id, score=score))
    
    return RecommendationResponse(
        user_id=user_id,
        items=items,
        total=len(items),
        cache_hit=False
    )

