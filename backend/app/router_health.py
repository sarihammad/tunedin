"""Health check and metrics router."""

from fastapi import APIRouter, Depends
from .deps import deps
from .schemas import HealthResponse
from .metrics import get_metrics

router = APIRouter(tags=["system"])


@router.get("/healthz", response_model=HealthResponse)
async def health_check(deps_instance=Depends(lambda: deps)):
    """Health check endpoint."""
    
    is_healthy = deps_instance.is_healthy()
    
    return HealthResponse(
        status="ok" if is_healthy else "degraded",
        version="1.0.0",
        uptime=deps_instance.get_uptime()
    )


@router.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return get_metrics()


@router.get("/status")
async def detailed_status(deps_instance=Depends(lambda: deps)):
    """Detailed system status."""
    
    status = {
        "healthy": deps_instance.is_healthy(),
        "uptime": deps_instance.get_uptime(),
        "dependencies": {
            "redis": deps_instance.redis_client is not None,
            "faiss_index": deps_instance.faiss_index is not None and deps_instance.faiss_index.index is not None,
            "user_embeddings": deps_instance.user_embeddings is not None,
            "item_embeddings": deps_instance.item_embeddings is not None,
        }
    }
    
    # Add FAISS index stats
    if deps_instance.faiss_index:
        status["faiss_stats"] = deps_instance.faiss_index.get_stats()
    
    # Add cache stats
    if deps_instance.cache_manager:
        status["cache_stats"] = deps_instance.cache_manager.get_cache_stats()
    
    return status

