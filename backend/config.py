"""
Configuration settings for the TunedIn project.

This file centralizes all the constants and environment configurations used
across the project—such as directory paths, model hyperparameters, API config,
dataset settings, and Spotify integration keys.
"""

import os
from pathlib import Path

# Base directory structure
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")  # Directory for raw data
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")  # Processed data files
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")  # Learned embeddings
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Saved model checkpoints

# API server settings
API_HOST = "0.0.0.0"  # Bind to all interfaces
API_PORT = 8000       # Port the FastAPI server will run on
API_WORKERS = 4       # Number of worker processes for the API server

# Model training hyperparameters
EMBEDDING_DIM = 128
NUM_GNN_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
TOP_K_RECOMMENDATIONS = 10  # Default number of top-K recommendations to return

# Node types in the graph
USER_NODE_TYPE = "user"
SONG_NODE_TYPE = "song"
ARTIST_NODE_TYPE = "artist"
ALBUM_NODE_TYPE = "album"
GENRE_NODE_TYPE = "genre"
PLAYLIST_NODE_TYPE = "playlist"

# Edge types in the graph
LISTENED_EDGE = "listened"
LIKED_EDGE = "liked"
SKIPPED_EDGE = "skipped"
PERFORMED_BY_EDGE = "performed_by"
BELONGS_TO_EDGE = "belongs_to"
PART_OF_EDGE = "part_of"
ADDED_TO_EDGE = "added_to"

# Dataset split ratios and negative sampling
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
NEGATIVE_SAMPLING_RATIO = 4  # Number of negatives per positive interaction

# Spotify API credentials (optional: used to generate real track features)
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")

# Evaluation metric names
METRICS = ["precision", "recall", "ndcg", "map", "hit_rate"]

# Features extracted from audio analysis
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

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(BASE_DIR, "tunedin.log")

# Ensure necessary directories exist at runtime
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    os.makedirs(directory, exist_ok=True)