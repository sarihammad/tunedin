# TunedIn Dataset

This directory contains the datasets used for training the TunedIn music recommendation system.

## Dataset Overview

### Last.fm Dataset
We use a subset of the Last.fm dataset for training and evaluation. This dataset contains:
- User-artist listening events
- Implicit feedback (play counts)
- Artist metadata and tags

### Data Structure

```
data/
├── raw/                    # Raw downloaded data
│   ├── user_artists.dat    # User-artist interaction matrix
│   ├── artists.dat         # Artist metadata
│   └── user_taggedartists.dat  # User-tagged artists
├── processed/              # Processed parquet files
│   ├── users.parquet       # User information
│   ├── tracks.parquet      # Track/artist information
│   ├── artists.parquet     # Artist metadata
│   └── user_events.parquet # User interaction events
└── scripts/                # Data processing scripts
    └── lastfm_download.py  # Download and process script
```

## Data Processing

The raw Last.fm data is processed into a format suitable for graph neural network training:

1. **User-Item Graph**: Bipartite graph with users and artists as nodes
2. **Edge Weights**: Normalized play counts as edge weights
3. **Temporal Split**: Train/validation/test splits based on timestamps
4. **Cold Start**: Separate handling for new users and artists

## Usage

To download and process the dataset:

```bash
cd ml
python -m data.scripts.lastfm_download
```

This will:
1. Download a sample of the Last.fm dataset
2. Process it into the required format
3. Save as parquet files for efficient loading

## Dataset Statistics

- **Users**: ~1,000 (sample)
- **Artists**: ~10,000 (sample)
- **Interactions**: ~100,000 (sample)
- **Sparsity**: ~99% (typical for recommendation datasets)

## Licensing

The Last.fm dataset is used under their research license. Please refer to the Last.fm website for full licensing terms.

## Alternative Datasets

For production use, consider:
- **Million Song Dataset (MSD)**: Larger scale, audio features
- **Spotify Million Playlist Dataset**: Playlist-based recommendations
- **Amazon Music**: Product-based recommendations

## Data Quality

- **Implicit Feedback**: Play counts as positive signals
- **Temporal Ordering**: Chronological interaction timestamps
- **User Filtering**: Minimum interaction thresholds
- **Artist Filtering**: Popularity-based filtering

