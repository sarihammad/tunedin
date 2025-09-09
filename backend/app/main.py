"""Main FastAPI application."""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from .deps import deps
from .metrics import PrometheusMiddleware
from .router_rec import router as rec_router
from .router_feedback import router as feedback_router
from .router_health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting TunedIn Music Recommender...")
    
    # Initialize dependencies
    success = deps.initialize()
    if not success:
        print("Warning: Some dependencies failed to initialize")
    
    yield
    
    # Shutdown
    print("Shutting down TunedIn Music Recommender...")


# Create FastAPI app
app = FastAPI(
    title="TunedIn Music Recommender",
    description="AI-powered music recommendation system with collaborative filtering and content-based features",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(rec_router)
app.include_router(feedback_router)
app.include_router(health_router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "TunedIn Music Recommender",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/healthz",
        "metrics": "/metrics"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("SERVICE_PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

