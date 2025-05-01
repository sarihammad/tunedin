"""
Data loading utilities for the TunedIn project.
Supports loading both real and mock datasets for development and testing.
"""
import os
import pandas as pd
import numpy as np
from loguru import logger
import requests
import zipfile
import io
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.config as config

# High-level function to load a dataset by its name.
# Decides which specific loader function to call based on the dataset_name parameter.
def load_dataset(dataset_name):
    """
    Load a dataset by name.
    
    Args:
        dataset_name (str): Name of the dataset to load.
            Options: 'spotify', 'lastfm', 'million_song'
            
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if dataset_name == "spotify":
        return load_spotify_dataset()
    elif dataset_name == "lastfm":
        return load_lastfm_dataset()
    elif dataset_name == "million_song":
        return load_million_song_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Loads the Spotify dataset.
# If the dataset file exists locally, it loads from disk.
# Otherwise, it attempts to create a sample dataset by querying the Spotify API.
# If API credentials are missing or an error occurs, it falls back to generating mock data.
def load_spotify_dataset():
    """
    Load the Spotify dataset.
    If the dataset doesn't exist locally, download a sample from Spotify API.
    
    Returns:
        pd.DataFrame: Spotify dataset with user-song interactions
    """
    spotify_file = os.path.join(config.PROCESSED_DATA_DIR, "spotify_dataset.csv")
    
    # Check if the dataset already exists locally
    if os.path.exists(spotify_file):
        logger.info(f"Loading Spotify dataset from {spotify_file}")
        return pd.read_csv(spotify_file)
    
    # Dataset file not found locally; create a sample dataset using Spotify API
    logger.info("Spotify dataset not found. Creating a sample dataset using Spotify API.")
    
    # Check if Spotify API credentials are configured
    if not config.SPOTIFY_CLIENT_ID or not config.SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify API credentials not found. Using mock data instead.")
        return create_mock_spotify_dataset()
    
    try:
        # Initialize the Spotify client with credentials
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=config.SPOTIFY_CLIENT_ID,
            client_secret=config.SPOTIFY_CLIENT_SECRET
        ))
        
        # Define genres to search tracks for
        genres = ["pop", "rock", "hip-hop", "electronic", "jazz", "classical"]
        tracks = []
        
        # Query Spotify API for top tracks in each genre
        for genre in genres:
            results = sp.search(q=f"genre:{genre}", type="track", limit=50)
            for item in results["tracks"]["items"]:
                track = {
                    "song_id": item["id"],
                    "song_name": item["name"],
                    "artist_id": item["artists"][0]["id"],
                    "artist_name": item["artists"][0]["name"],
                    "album_id": item["album"]["id"],
                    "album_name": item["album"]["name"],
                    "popularity": item["popularity"],
                    "genre": genre
                }
                tracks.append(track)
        
        # Convert track info into a DataFrame
        tracks_df = pd.DataFrame(tracks)
        
        # Retrieve audio features for tracks in batches of 100
        audio_features = []
        for i in range(0, len(tracks_df), 100):
            batch = tracks_df["song_id"][i:i+100].tolist()
            features = sp.audio_features(batch)
            audio_features.extend(features)
        
        # Create DataFrame of audio features
        audio_features_df = pd.DataFrame(audio_features)
        
        # Merge track info with audio features on song ID
        songs_df = pd.merge(
            tracks_df,
            audio_features_df[["id"] + config.AUDIO_FEATURES],
            left_on="song_id",
            right_on="id"
        ).drop(columns=["id"])
        
        # Generate synthetic user interactions with songs
        num_users = 1000
        num_interactions_per_user = 50
        
        user_interactions = []
        
        for user_id in range(num_users):
            # Randomly select songs listened to by this user
            user_songs = np.random.choice(
                songs_df["song_id"].values,
                size=num_interactions_per_user,
                replace=False
            )
            
            for song_id in user_songs:
                # Randomly assign interaction type: listened, liked, or skipped
                interaction_type = np.random.choice(
                    [config.LISTENED_EDGE, config.LIKED_EDGE, config.SKIPPED_EDGE],
                    p=[0.7, 0.2, 0.1]
                )
                
                # Assign interaction count based on type
                if interaction_type == config.LISTENED_EDGE:
                    interaction_count = np.random.randint(1, 20)
                elif interaction_type == config.LIKED_EDGE:
                    interaction_count = 1
                else:  # skipped
                    interaction_count = 1
                
                user_interactions.append({
                    "user_id": f"user_{user_id}",
                    "song_id": song_id,
                    "interaction_type": interaction_type,
                    "interaction_count": interaction_count,
                    "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
                })
        
        # Create DataFrame of user interactions
        interactions_df = pd.DataFrame(user_interactions)
        
        # Save songs and interactions datasets separately
        songs_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "spotify_songs.csv"), index=False)
        interactions_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "spotify_interactions.csv"), index=False)
        
        # Combine datasets into one dictionary for ease of use
        dataset = {
            "songs": songs_df,
            "interactions": interactions_df
        }
        
        # Save combined dataset as a pickle file
        pd.to_pickle(dataset, spotify_file.replace(".csv", ".pkl"))
        
        return dataset
        
    except Exception as e:
        logger.error(f"Error creating Spotify dataset: {e}")
        logger.info("Using mock data instead.")
        return create_mock_spotify_dataset()

# Creates a mock Spotify dataset for testing and development.
# Generates synthetic songs with audio features and user interactions.
def create_mock_spotify_dataset():
    """
    Create a mock Spotify dataset for testing purposes.
    
    Returns:
        pd.DataFrame: Mock Spotify dataset
    """
    # --------------------- Generate mock songs ---------------------
    num_songs = 1000
    num_artists = 100
    num_albums = 200
    num_genres = 10
    
    # Generate song, artist, and album IDs
    song_ids = [f"song_{i}" for i in range(num_songs)]
    artist_ids = [f"artist_{i}" for i in range(num_artists)]
    album_ids = [f"album_{i}" for i in range(num_albums)]
    
    # Define possible genres
    genres = ["pop", "rock", "hip-hop", "electronic", "jazz", "classical", "country", "r&b", "metal", "folk"]
    
    # Create list to hold song dictionaries
    songs = []
    for i in range(num_songs):
        song = {
            "song_id": song_ids[i],
            "song_name": f"Song {i}",
            "artist_id": np.random.choice(artist_ids),
            "artist_name": f"Artist {np.random.randint(0, num_artists)}",
            "album_id": np.random.choice(album_ids),
            "album_name": f"Album {np.random.randint(0, num_albums)}",
            "popularity": np.random.randint(0, 100),
            "genre": np.random.choice(genres)
        }
        
        # --------------------- Add audio features ---------------------
        for feature in config.AUDIO_FEATURES:
            if feature in ["key", "mode"]:
                song[feature] = np.random.randint(0, 12)
            elif feature in ["tempo"]:
                song[feature] = np.random.uniform(60, 200)
            elif feature in ["loudness"]:
                song[feature] = np.random.uniform(-20, 0)
            else:
                song[feature] = np.random.uniform(0, 1)
        
        songs.append(song)
    
    songs_df = pd.DataFrame(songs)
    
    # --------------------- Generate synthetic user interactions ---------------------
    num_users = 1000
    num_interactions_per_user = 50
    
    user_interactions = []
    
    for user_id in range(num_users):
        # Randomly select songs for this user
        user_songs = np.random.choice(
            song_ids,
            size=num_interactions_per_user,
            replace=False
        )
        
        for song_id in user_songs:
            # Randomly assign interaction type
            interaction_type = np.random.choice(
                [config.LISTENED_EDGE, config.LIKED_EDGE, config.SKIPPED_EDGE],
                p=[0.7, 0.2, 0.1]
            )
            
            # Assign interaction count based on type
            if interaction_type == config.LISTENED_EDGE:
                interaction_count = np.random.randint(1, 20)
            elif interaction_type == config.LIKED_EDGE:
                interaction_count = 1
            else:  # skipped
                interaction_count = 1
            
            user_interactions.append({
                "user_id": f"user_{user_id}",
                "song_id": song_id,
                "interaction_type": interaction_type,
                "interaction_count": interaction_count,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365))
            })
    
    interactions_df = pd.DataFrame(user_interactions)
    
    # --------------------- Save datasets ---------------------
    songs_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "spotify_songs.csv"), index=False)
    interactions_df.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "spotify_interactions.csv"), index=False)
    
    dataset = {
        "songs": songs_df,
        "interactions": interactions_df
    }
    
    pd.to_pickle(dataset, os.path.join(config.PROCESSED_DATA_DIR, "spotify_dataset.pkl"))
    
    return dataset

# Loads the Last.fm dataset.
# If the dataset is not present locally, it currently uses a mock dataset instead of downloading the actual data,
# to facilitate development without requiring large downloads.
def load_lastfm_dataset():
    """
    Load the Last.fm dataset.
    If the dataset doesn't exist locally, download it from the source.
    
    Returns:
        pd.DataFrame: Last.fm dataset with user-song interactions
    """
    lastfm_file = os.path.join(config.PROCESSED_DATA_DIR, "lastfm_dataset.pkl")
    
    # Check if the dataset already exists locally
    if os.path.exists(lastfm_file):
        logger.info(f"Loading Last.fm dataset from {lastfm_file}")
        return pd.read_pickle(lastfm_file)
    
    # Dataset not found locally; in development, use mock data instead of downloading
    logger.info("Last.fm dataset not found. Downloading...")
    
    # URL for the Last.fm 1K dataset (not used here, placeholder)
    url = "http://mtg.upf.edu/static/datasets/last.fm/lastfm-dataset-1K.tar.gz"
    
    try:
        # Using mock data for development purposes instead of actual download
        logger.info("Using mock Last.fm data for development purposes.")
        return create_mock_lastfm_dataset()
    except Exception as e:
        logger.error(f"Error downloading Last.fm dataset: {e}")
        logger.info("Using mock data instead.")
        return create_mock_lastfm_dataset()

# Creates a mock Last.fm dataset.
# For now, reuses the mock Spotify dataset structure as a placeholder.
def create_mock_lastfm_dataset():
    """
    Create a mock Last.fm dataset for testing purposes.
    
    Returns:
        pd.DataFrame: Mock Last.fm dataset
    """
    # Similar to the mock Spotify dataset but with Last.fm specific fields
    return create_mock_spotify_dataset()  # Reuse the same mock data structure for now

# Loads the Million Song Dataset.
# If the dataset is not present locally, uses a mock dataset instead of downloading a large subset,
# to facilitate development and testing.
def load_million_song_dataset():
    """
    Load the Million Song Dataset.
    If the dataset doesn't exist locally, download a subset.
    
    Returns:
        pd.DataFrame: Million Song Dataset with song features
    """
    msd_file = os.path.join(config.PROCESSED_DATA_DIR, "msd_dataset.pkl")
    
    # Check if the dataset already exists locally
    if os.path.exists(msd_file):
        logger.info(f"Loading Million Song Dataset from {msd_file}")
        return pd.read_pickle(msd_file)
    
    # Dataset not found locally; using mock data instead of actual download
    logger.info("Million Song Dataset not found. Using mock data.")
    return create_mock_million_song_dataset()

# Creates a mock Million Song Dataset.
# Currently reuses the mock Spotify dataset structure as a placeholder.
def create_mock_million_song_dataset():
    """
    Create a mock Million Song Dataset for testing purposes.
    
    Returns:
        pd.DataFrame: Mock Million Song Dataset
    """
    # Similar structure to the mock Spotify dataset
    return create_mock_spotify_dataset()  # Reuse the same mock data structure for now

# Downloads a file from a URL and saves it to a specified path.
def download_file(url, save_path):
    """
    Download a file from a URL and save it to a path.
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the file to
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded {url} to {save_path}")

# Extracts a zip file to a given directory.
def extract_zip(zip_path, extract_to):
    """
    Extract a zip file to a directory.
    
    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory to extract to
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Extracted {zip_path} to {extract_to}")