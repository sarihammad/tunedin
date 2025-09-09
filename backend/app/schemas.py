"""Pydantic schemas for API requests and responses."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class RecommendationItem(BaseModel):
    """Individual recommendation item."""
    track_id: str = Field(..., description="Unique track identifier")
    score: float = Field(..., description="Recommendation score (0-1)")


class RecommendationResponse(BaseModel):
    """Response for recommendation requests."""
    user_id: str = Field(..., description="User identifier")
    items: List[RecommendationItem] = Field(..., description="List of recommended tracks")
    total: int = Field(..., description="Total number of recommendations")
    cache_hit: bool = Field(..., description="Whether user embedding was cached")


class FeedbackRequest(BaseModel):
    """Request for user feedback submission."""
    user_id: str = Field(..., description="User identifier")
    track_id: str = Field(..., description="Track identifier")
    event: Literal["play", "like", "skip"] = Field(..., description="User interaction type")
    ts: int = Field(..., description="Unix timestamp of interaction")


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    success: bool = Field(..., description="Whether feedback was recorded")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

