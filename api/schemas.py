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