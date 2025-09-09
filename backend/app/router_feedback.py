"""Feedback router for user interaction tracking."""

from fastapi import APIRouter, HTTPException, Depends
from .deps import deps
from .schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    deps_instance=Depends(lambda: deps)
):
    """Submit user feedback (play, like, skip)."""
    
    try:
        # Log the feedback (in production, this would go to Kafka or a database)
        print(f"Feedback received: user={feedback.user_id}, track={feedback.track_id}, "
              f"event={feedback.event}, ts={feedback.ts}")
        
        # Add to recent items to avoid repeats in recommendations
        if deps_instance.cache_manager:
            deps_instance.cache_manager.add_recent_item(
                feedback.user_id, 
                feedback.track_id
            )
        
        # In a production system, you would:
        # 1. Validate the feedback data
        # 2. Store in a time-series database
        # 3. Send to a message queue for real-time processing
        # 4. Update user embeddings incrementally
        # 5. Trigger model retraining if needed
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded for {feedback.event} event"
        )
        
    except Exception as e:
        print(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to process feedback")


@router.get("/stats")
async def get_feedback_stats(deps_instance=Depends(lambda: deps)):
    """Get feedback statistics (placeholder)."""
    
    # In production, this would query actual feedback statistics
    return {
        "total_feedback": 0,
        "feedback_by_type": {
            "play": 0,
            "like": 0,
            "skip": 0
        },
        "recent_activity": []
    }

