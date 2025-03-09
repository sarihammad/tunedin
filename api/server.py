"""
API server for the TunedIn recommendation system.
"""
import os
import sys
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel
from utils.data_loader import load_dataset
from utils.graph_builder import build_graph
from loguru import logger

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

# Global variables
models = {}
graph = None
song_id_to_idx = {}
idx_to_song_id = {}
song_metadata = {}

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 10
    exclude_listened: bool = True

class SongRecommendationRequest(BaseModel):
    song_id: str
    num_recommendations: int = 10

class UserFeatureRequest(BaseModel):
    features: Dict[str, float]
    num_recommendations: int = 10

class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    model_used: str

class ModelInfoResponse(BaseModel):
    name: str
    is_trained: bool
    embedding_dim: int
    num_layers: int

class StatusResponse(BaseModel):
    status: str
    models_loaded: List[str]
    num_users: int
    num_songs: int

@app.on_event("startup")
async def startup_event():
    """
    Load models and data on startup.
    """
    global models, graph, song_id_to_idx, idx_to_song_id, song_metadata
    
    logger.info("Starting TunedIn API server")
    
    # Load dataset
    try:
        logger.info("Loading dataset")
        dataset = load_dataset("spotify")
        
        # Build graph
        logger.info("Building graph")
        graph = build_graph(dataset)
        
        # Create song ID mappings
        songs_df = dataset["songs"]
        song_id_to_idx = {song_id: i for i, song_id in enumerate(songs_df["song_id"])}
        idx_to_song_id = {i: song_id for song_id, i in song_id_to_idx.items()}
        
        # Create song metadata
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
        
        # Load models
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
        
        logger.info("API server startup complete")
    
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue without models - they can be trained via the API

@app.get("/", response_model=StatusResponse)
async def root():
    """
    Get API status.
    """
    return {
        "status": "online",
        "models_loaded": [name for name, model in models.items() if model.is_trained],
        "num_users": graph[config.USER_NODE_TYPE].num_nodes if graph else 0,
        "num_songs": graph[config.SONG_NODE_TYPE].num_nodes if graph else 0
    }

@app.get("/models", response_model=Dict[str, ModelInfoResponse])
async def get_models():
    """
    Get information about available models.
    """
    return {
        name: {
            "name": name,
            "is_trained": model.is_trained,
            "embedding_dim": model.embedding_dim,
            "num_layers": getattr(model, "num_layers", config.NUM_GNN_LAYERS)
        }
        for name, model in models.items()
    }

@app.post("/recommend/user", response_model=RecommendationResponse)
async def recommend_for_user(
    request: RecommendationRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations")
):
    """
    Get recommendations for a user.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = models[model_name]
    
    if not model.is_trained:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")
    
    try:
        # Get user index
        user_id = request.user_id
        
        # Check if user exists in the graph
        if not user_id.startswith("user_"):
            user_id = f"user_{user_id}"
        
        # Get user interactions
        edge_index = graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index
        
        # Find user index
        user_indices = (graph[config.USER_NODE_TYPE].x == torch.eye(graph[config.USER_NODE_TYPE].num_nodes)[int(user_id.split("_")[1])]).all(dim=1).nonzero().squeeze()
        
        if user_indices.dim() == 0:
            user_indices = user_indices.unsqueeze(0)
        
        if user_indices.size(0) == 0:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")
        
        user_idx = user_indices[0].item()
        
        # Get user interactions
        user_interactions = {}
        for i in range(edge_index.size(1)):
            idx = edge_index[0, i].item()
            item_idx = edge_index[1, i].item()
            
            if idx not in user_interactions:
                user_interactions[idx] = []
            
            user_interactions[idx].append(item_idx)
        
        # Generate recommendations
        recommendations = model.recommend(
            user_indices=torch.tensor([user_idx]),
            top_k=request.num_recommendations,
            exclude_interacted=request.exclude_listened,
            interacted_items=user_interactions if request.exclude_listened else None
        )[0]
        
        # Convert to song IDs and add metadata
        recommendation_items = []
        for item_idx in recommendations:
            song_id = idx_to_song_id.get(item_idx)
            if song_id and song_id in song_metadata:
                recommendation_items.append(song_metadata[song_id])
        
        return {
            "recommendations": recommendation_items,
            "model_used": model_name
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/song", response_model=RecommendationResponse)
async def recommend_similar_songs(
    request: SongRecommendationRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations")
):
    """
    Get recommendations for similar songs.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = models[model_name]
    
    if not model.is_trained:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")
    
    try:
        # Get song index
        song_id = request.song_id
        
        if song_id not in song_id_to_idx:
            raise HTTPException(status_code=404, detail=f"Song {song_id} not found")
        
        song_idx = song_id_to_idx[song_id]
        
        # Get song embedding
        song_emb = model.item_embedding[song_idx].unsqueeze(0)
        
        # Calculate similarity with all songs
        all_song_emb = model.item_embedding
        similarities = torch.matmul(song_emb, all_song_emb.t()).squeeze()
        
        # Get top-k similar songs (excluding the query song)
        similarities[song_idx] = -float('inf')
        top_indices = torch.topk(similarities, k=request.num_recommendations).indices.tolist()
        
        # Convert to song IDs and add metadata
        recommendation_items = []
        for item_idx in top_indices:
            song_id = idx_to_song_id.get(item_idx)
            if song_id and song_id in song_metadata:
                recommendation_items.append(song_metadata[song_id])
        
        return {
            "recommendations": recommendation_items,
            "model_used": model_name
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/features", response_model=RecommendationResponse)
async def recommend_from_features(
    request: UserFeatureRequest,
    model_name: str = Query("graphsage", description="Model to use for recommendations")
):
    """
    Get recommendations based on user features (for cold-start users).
    This is particularly useful for GraphSAGE which supports inductive learning.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    model = models[model_name]
    
    if not model.is_trained:
        raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")
    
    if model_name != "graphsage":
        raise HTTPException(status_code=400, detail="Only GraphSAGE model supports inductive learning for cold-start users")
    
    try:
        # Create feature vector
        feature_vector = torch.zeros(config.EMBEDDING_DIM)
        
        # Fill in provided features
        for i, (feature, value) in enumerate(request.features.items()):
            if i < config.EMBEDDING_DIM:
                feature_vector[i] = value
        
        # Reshape for batch processing
        user_features = feature_vector.unsqueeze(0)
        
        # Get all item features
        item_features = model.item_embedding
        
        # Generate recommendations using inductive learning
        recommendations = model.inductive_recommend(
            user_features=user_features,
            item_features=item_features,
            top_k=request.num_recommendations
        )[0]
        
        # Convert to song IDs and add metadata
        recommendation_items = []
        for item_idx in recommendations:
            song_id = idx_to_song_id.get(item_idx)
            if song_id and song_id in song_metadata:
                recommendation_items.append(song_metadata[song_id])
        
        return {
            "recommendations": recommendation_items,
            "model_used": model_name
        }
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train/{model_name}")
async def train_model(
    model_name: str,
    epochs: int = Query(10, description="Number of training epochs"),
    learning_rate: float = Query(0.001, description="Learning rate")
):
    """
    Train a model.
    """
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    if graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    
    try:
        logger.info(f"Training {model_name} model for {epochs} epochs")
        
        # Train model
        model = models[model_name]
        history = model.train(
            graph=graph,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Save model
        model_path = os.path.join(config.MODELS_DIR, f"{model_name}_spotify.pt")
        model.save(model_path)
        
        return {
            "status": "success",
            "model": model_name,
            "epochs": epochs,
            "final_loss": history["loss"][-1] if history["loss"] else None
        }
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    """
    Start the API server.
    """
    uvicorn.run(
        "api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        reload=True
    )

if __name__ == "__main__":
    start_server() 