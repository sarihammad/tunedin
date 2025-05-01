class RecommendationRequest(BaseModel):
    """Request schema for user-based recommendations."""
    user_id: str  # ID of the user requesting recommendations
    num_recommendations: int = 10  # Number of recommendations to return
    exclude_listened: bool = True  # Whether to exclude songs the user has already listened to

class SongRecommendationRequest(BaseModel):
    """Request schema for recommending songs similar to a given song."""
    song_id: str  # ID of the reference song
    num_recommendations: int = 10  # Number of similar songs to recommend

class UserFeatureRequest(BaseModel):
    """Request schema for cold-start recommendations based on user-provided audio features."""
    features: Dict[str, float]  # Dictionary of audio features (e.g., tempo, energy)
    num_recommendations: int = 10  # Number of recommendations to return

class RecommendationResponse(BaseModel):
    """Response schema containing recommended songs and model metadata."""
    recommendations: List[Dict[str, Any]]  # List of recommended songs
    model_used: str  # Name of the model that generated the recommendations

class ModelInfoResponse(BaseModel):
    """Response schema containing information about a trained model."""
    name: str  # Name of the model
    is_trained: bool  # Whether the model has been trained
    embedding_dim: int  # Size of the embedding vectors used in the model
    num_layers: int  # Number of GNN layers in the model

class StatusResponse(BaseModel):
    """Response schema for the API status endpoint."""
    status: str  # Overall API status ("online", "offline")
    models_loaded: List[str]  # List of currently loaded models
    num_users: int  # Number of users in the dataset
    num_songs: int  # Number of songs in the dataset