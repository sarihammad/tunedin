import pytest
from fastapi.testclient import TestClient
from api.server import app
from api.services.recommendation_service import RecommendationService

# Override the dependency to avoid loading models from disk
app.dependency_overrides[RecommendationService] = lambda: RecommendationService()

client = TestClient(app)

def test_recommend_for_user():
    response = client.post(
        "/recommend/user?model_name=graphsage",
        json={
            "user_id": "user_0",
            "num_recommendations": 5,
            "exclude_listened": True
        }
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_recommend_similar_songs():
    response = client.post(
        "/recommend/song?model_name=graphsage",
        json={
            "song_id": "song_0",
            "num_recommendations": 5
        }
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_recommend_from_features():
    response = client.post(
        "/recommend/features?model_name=graphsage",
        json={
            "features": {
                "danceability": 0.5,
                "energy": 0.7,
                "speechiness": 0.1,
                "acousticness": 0.2,
                "instrumentalness": 0.0,
                "liveness": 0.3,
                "valence": 0.6,
                "tempo": 120.0,
                "loudness": -6.0
            },
            "num_recommendations": 5
        }
    )
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_train_model():
    response = client.post(
        "/train/graphsage?epochs=1&learning_rate=0.001"
    )
    assert response.status_code == 200
    assert "message" in response.json()
