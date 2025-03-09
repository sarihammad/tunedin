# TunedIn: AI-Powered Music Recommendation System

TunedIn is an AI music recommendation system that personalizes song suggestions using Graph Neural Networks (GNNs), user listening habits, and audio feature analysis to create a seamless and intelligent music discovery experience.

## Project Objective

TunedIn is a next-gen music recommendation system that:

- Understands user preferences through Graph Neural Networks (GNNs)
- Learns from listening history, genres, mood, and song characteristics
- Provides intelligent, real-time recommendations and dynamic playlist generation
- Handles the cold-start problem for new users and new songs
- Scales to millions of users and tracks efficiently

## Project Structure

```
tunedin/
├── data/                  # Data storage and processing
│   ├── raw/               # Raw datasets
│   ├── processed/         # Processed datasets
│   └── embeddings/        # Pre-computed embeddings
├── models/                # GNN model implementations
│   ├── gcn.py             # Graph Convolutional Network
│   ├── gat.py             # Graph Attention Network
│   ├── lightgcn.py        # LightGCN implementation
│   └── graphsage.py       # GraphSAGE implementation
├── utils/                 # Utility functions
│   ├── data_loader.py     # Data loading utilities
│   ├── graph_builder.py   # Graph construction utilities
│   └── evaluation.py      # Model evaluation metrics
├── api/                   # API implementation
│   ├── routes/            # API endpoints
│   └── server.py          # API server
├── web/                   # Web interface (optional)
├── notebooks/             # Jupyter notebooks for exploration
├── tests/                 # Unit and integration tests
├── config.py              # Configuration settings
├── main.py                # Main entry point
└── requirements.txt       # Project dependencies
```

## Tech Stack

- **Graph Representation**: NetworkX, Neo4j, PyTorch-Geometric
- **GNN Models**: GCN, GraphSAGE, LightGCN
- **Embedding Storage**: FAISS, ChromaDB
- **Audio Feature Extraction**: Librosa, Spotify API
- **Backend API**: Python, FastAPI
- **Frontend** (Optional): Streamlit, React
- **Deployment**: Docker, Kubernetes

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NetworkX
- FastAPI

### Installation

1. Clone the repository:

```bash
git clone https://github.com/sarihammad/tunedin.git
cd tunedin
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

## Datasets

- **Million Song Dataset**: Large-scale song metadata and audio features
- **Spotify API / Last.fm API**: Real-time user listening history and song metadata
- **MusicBrainz Dataset**: Knowledge graph of artists, albums, and relationships

## License

This project is licensed under the MIT License - see the LICENSE file for details.
