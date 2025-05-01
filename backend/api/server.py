"""
API server for the TunedIn recommendation system.

This module defines the FastAPI application, loads data and models on startup,
and exposes endpoints for recommendation, model training, and model metadata access.
"""
# Standard Library
import os
import sys

from api.middleware import APIKeyMiddleware

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Third-Party Libraries
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime
from loguru import logger

# Local Application Imports
import backend.config as config
from api.routes import router as api_router
from api.services.recommendation_service import RecommendationService
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel
from schemas import *
from utils.data_loader import load_dataset
from utils.graph_builder import build_graph
from utils.model_loader import load_model
from datetime import datetime
from utils.model_reloader import watch_model_files
from api.dependencies import verify_admin

# model_train_times = {}  # in-memory store

# Create FastAPI app
app = FastAPI(
    title="TunedIn API",
    description="API for the TunedIn music recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(APIKeyMiddleware) 

app.include_router(api_router, prefix="/api")

# Dependency function to inject a fresh instance of RecommendationService
def get_recommendation_service():
    return RecommendationService()


@app.on_event("startup")
async def startup_event():
    """
    Load models and data on startup.
    """
    global models, graph, song_id_to_idx, idx_to_song_id, song_metadata
    
    logger.info("Starting TunedIn API server")
    
    # Step 1: Load dataset from disk or API
    try:
        logger.info("Loading dataset")
        dataset = load_dataset("spotify")
        
        # Step 2: Build graph structure from dataset
        logger.info("Building graph")
        graph = build_graph(dataset)
        
        # Step 3: Generate mappings for song metadata
        songs_df = dataset["songs"]
        song_id_to_idx = {song_id: i for i, song_id in enumerate(songs_df["song_id"])}
        idx_to_song_id = {i: song_id for song_id, i in song_id_to_idx.items()}
        
        # Construct a metadata dictionary for each song (ID, name, artist, features)
        song_metadata = {}
        for _, row in songs_df.iterrows():
            song_id = row["song_id"]
            song_metadata[song_id] = {
                "song_id": song_id,
                "song_name": row["song_name"],
                "artist_id": row["artist_id"],
                "artist_name": row["artist_name"],
                "album_id": row["album_id"],
                "album_name": row["album_name"],
                "genre": row["genre"],
                "popularity": int(row["popularity"]) if "popularity" in row else None
            }
            
            # Add audio features
            for feature in config.AUDIO_FEATURES:
                if feature in row:
                    song_metadata[song_id][feature] = float(row[feature])
        
        # Step 4: Load or initialize models
        logger.info("Loading models")
        model_classes = {
            "gcn": GCNModel,
            "gat": GATModel,
            "lightgcn": LightGCNModel,
            "graphsage": GraphSAGEModel
        }
        
        for model_name, model_class in model_classes.items():
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_spotify.pt")
            
            if os.path.exists(model_path):
                logger.info(f"Loading {model_name} model from {model_path}")
                models[model_name] = model_class.load(model_path)
            else:
                logger.info(f"Model {model_name} not found at {model_path}")
                # Initialize untrained model
                models[model_name] = model_class(
                    input_dim=None,
                    embedding_dim=config.EMBEDDING_DIM,
                    num_layers=config.NUM_GNN_LAYERS
                )
        
        watch_model_files(models, config.MODELS_DIR)
        logger.info("API server startup complete")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue without models - they can be trained via the API

@app.get("/", response_model=StatusResponse)
async def root(service: RecommendationService = Depends(get_recommendation_service)):
    # Endpoint to get API status and basic info
    return {
        "status": "online",
        "models_loaded": [k for k, m in service.models.items() if m.is_trained],
        "num_users": service.graph[config.USER_NODE_TYPE].num_nodes,
        "num_songs": service.graph[config.SONG_NODE_TYPE].num_nodes
    }

@app.get("/models", response_model=Dict[str, ModelInfoResponse])
async def get_models(service: RecommendationService = Depends(get_recommendation_service)):
    """
    Get information about available models.
    """
    # Endpoint to get metadata about all available models
    return {
        name: {
            "name": name,
            "is_trained": model.is_trained,
            "embedding_dim": model.embedding_dim,
            "num_layers": getattr(model, "num_layers", config.NUM_GNN_LAYERS)
        }
        for name, model in service.models.items()
    }


@app.post("/recommend/user", response_model=RecommendationResponse)
async def recommend_for_user(
    request: RecommendationRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    # Endpoint to get personalized song recommendations for a given user
    return await service.recommend_for_user(request, model_name)


@app.post("/recommend/song", response_model=RecommendationResponse)
async def recommend_similar_songs(
    request: SongRecommendationRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    # Endpoint to get song recommendations similar to a given song
    return await service.recommend_similar_songs(request, model_name)

@app.post("/recommend/features", response_model=RecommendationResponse)
async def recommend_from_features(
    request: UserFeatureRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    # Endpoint to get recommendations based on user features
    return await service.recommend_from_features(request, model_name)


@app.post("/train/{model_name}", dependencies=[Depends(verify_admin)])
async def train_model(
    model_name: str,
    epochs: int = Query(10, description="Number of training epochs"),
    learning_rate: float = Query(0.001, description="Learning rate"),
    service: RecommendationService = Depends(get_recommendation_service)
):
    # Endpoint to train a specified model with given parameters
    result = await service.train_model(model_name, epochs, learning_rate)
    service.models[model_name].loss_history = result.get("loss", [])
    service.models[model_name].last_trained = datetime.utcnow().isoformat()
    return result

@app.get("/status/{model_name}", response_model=Dict[str, Any])
async def get_model_status(
    model_name: str,
    service: RecommendationService = Depends(get_recommendation_service)
):
    model = service.models.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    recommendations = []
    try:
        recommendations = model.recommend(user_indices=torch.tensor([0]), top_k=5)
    except:
        pass  # model may not be trained or user_0 doesn't exist
    
    return {
        "model_name": model_name,
        "is_trained": model.is_trained,
        "last_trained": getattr(model, "last_trained", None),
        "training_loss_history": getattr(model, "loss_history", []),
        "sample_recommendations": recommendations[0] if recommendations else []
    }

def start_server():
    """
    Start the API server.
    """
    # Start Uvicorn server in development mode with auto-reload enabled
    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=True
        # Set to false for production
    )

if __name__ == "__main__":
    start_server()