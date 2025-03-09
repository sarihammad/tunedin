"""
Configuration settings for the TunedIn project.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Model settings
EMBEDDING_DIM = 128
NUM_GNN_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
TOP_K_RECOMMENDATIONS = 10

# Graph settings
USER_NODE_TYPE = "user"
SONG_NODE_TYPE = "song"
ARTIST_NODE_TYPE = "artist"
ALBUM_NODE_TYPE = "album"
GENRE_NODE_TYPE = "genre"
PLAYLIST_NODE_TYPE = "playlist"

# Edge types
LISTENED_EDGE = "listened"
LIKED_EDGE = "liked"
SKIPPED_EDGE = "skipped"
PERFORMED_BY_EDGE = "performed_by"
BELONGS_TO_EDGE = "belongs_to"
PART_OF_EDGE = "part_of"
ADDED_TO_EDGE = "added_to"

# Dataset settings
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
NEGATIVE_SAMPLING_RATIO = 4

# Spotify API settings (to be filled with actual credentials)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Evaluation metrics
METRICS = ["precision", "recall", "ndcg", "map", "hit_rate"]

# Feature extraction settings
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "tunedin.log")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    os.makedirs(directory, exist_ok=True) 