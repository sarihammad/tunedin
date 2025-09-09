"""Download and process Last.fm dataset for training."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile


def download_file(url: str, filename: str) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            size = file.write(chunk)
            progress_bar.update(size)


def create_sample_data() -> None:
    """Create sample Last.fm-like data for demonstration."""
    
    # Create directories
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating sample Last.fm dataset...")
    
    # Generate sample data
    np.random.seed(42)
    
    # Users
    n_users = 1000
    n_artists = 10000
    n_interactions = 50000
    
    # Create user-artist interactions
    user_ids = np.random.randint(0, n_users, n_interactions)
    artist_ids = np.random.randint(0, n_artists, n_interactions)
    play_counts = np.random.poisson(3, n_interactions) + 1  # Minimum 1 play
    
    # Create timestamps (last 2 years)
    timestamps = np.random.randint(
        int(pd.Timestamp.now().timestamp()) - 2 * 365 * 24 * 3600,
        int(pd.Timestamp.now().timestamp()),
        n_interactions
    )
    
    # Create DataFrames
    user_events = pd.DataFrame({
        'user_id': user_ids,
        'artist_id': artist_ids,
        'play_count': play_counts,
        'timestamp': timestamps
    })
    
    # Remove duplicates and aggregate
    user_events = user_events.groupby(['user_id', 'artist_id']).agg({
        'play_count': 'sum',
        'timestamp': 'max'
    }).reset_index()
    
    # Create user metadata
    users = pd.DataFrame({
        'user_id': range(n_users),
        'age': np.random.randint(18, 65, n_users),
        'country': np.random.choice(['US', 'UK', 'CA', 'DE', 'FR'], n_users),
        'gender': np.random.choice(['M', 'F', 'O'], n_users, p=[0.5, 0.4, 0.1])
    })
    
    # Create artist metadata
    artists = pd.DataFrame({
        'artist_id': range(n_artists),
        'artist_name': [f'Artist_{i}' for i in range(n_artists)],
        'genre': np.random.choice([
            'Rock', 'Pop', 'Hip-Hop', 'Electronic', 'Jazz', 'Classical', 
            'Country', 'R&B', 'Reggae', 'Blues'
        ], n_artists),
        'popularity': np.random.exponential(1, n_artists)
    })
    
    # Create tracks (simplified: one track per artist)
    tracks = pd.DataFrame({
        'track_id': range(n_artists),
        'artist_id': range(n_artists),
        'track_name': [f'Track_{i}' for i in range(n_artists)],
        'duration': np.random.randint(120, 300, n_artists)  # 2-5 minutes
    })
    
    # Save raw data
    print("Saving raw data...")
    user_events.to_csv(raw_dir / "user_artists.dat", sep='\t', index=False)
    users.to_csv(raw_dir / "users.dat", sep='\t', index=False)
    artists.to_csv(raw_dir / "artists.dat", sep='\t', index=False)
    tracks.to_csv(raw_dir / "tracks.dat", sep='\t', index=False)
    
    # Save processed data
    print("Saving processed data...")
    user_events.to_parquet(processed_dir / "user_events.parquet", index=False)
    users.to_parquet(processed_dir / "users.parquet", index=False)
    artists.to_parquet(processed_dir / "artists.parquet", index=False)
    tracks.to_parquet(processed_dir / "tracks.parquet", index=False)
    
    print(f"Dataset created successfully!")
    print(f"Users: {len(users)}")
    print(f"Artists: {len(artists)}")
    print(f"Tracks: {len(tracks)}")
    print(f"Interactions: {len(user_events)}")
    print(f"Sparsity: {1 - len(user_events) / (len(users) * len(artists)):.4f}")


def main():
    """Main function to download and process dataset."""
    print("TunedIn Dataset Downloader")
    print("=" * 40)
    
    # For demo purposes, create sample data instead of downloading real Last.fm data
    # In production, you would download from the actual Last.fm dataset
    create_sample_data()
    
    print("\nDataset ready for training!")


if __name__ == "__main__":
    main()

