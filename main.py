"""
TunedIn: AI-Powered Music Recommendation System
Main entry point for the application.
"""
import argparse
import os
import sys
from loguru import logger

import config
from utils.data_loader import load_dataset
from utils.graph_builder import build_graph
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    level=config.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(config.LOG_FILE, rotation="10 MB", level=config.LOG_LEVEL)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="TunedIn: AI-Powered Music Recommendation System")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "evaluate", "serve"],
                        help="Mode to run the application in")
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "lightgcn", "graphsage"],
                        help="GNN model to use")
    parser.add_argument("--dataset", type=str, default="spotify",
                        help="Dataset to use (spotify, lastfm, or million_song)")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=config.EMBEDDING_DIM,
                        help="Dimension of embeddings")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="Learning rate")
    
    return parser.parse_args()

def log_graph_stats(graph):
    """Compute and log the total number of nodes and edges in the heterogeneous graph."""
    total_nodes = 0
    # Sum node counts from each node type defined in config
    for node_type in [config.USER_NODE_TYPE, config.SONG_NODE_TYPE, config.ARTIST_NODE_TYPE,
                      config.ALBUM_NODE_TYPE, config.GENRE_NODE_TYPE]:
        if node_type in graph:
            total_nodes += graph[node_type].num_nodes
    total_edges = 0
    # Loop through all edge types in the hetero graph metadata
    for src, rel, dst in graph.metadata()[1]:
        if (src, rel, dst) in graph:
            total_edges += graph[(src, rel, dst)].edge_index.size(1)
    logger.info(f"Built graph with {total_nodes} nodes and {total_edges} edges")

def train(args):
    """Train the GNN model."""
    logger.info(f"Training {args.model} model on {args.dataset} dataset")
    
    # Load dataset
    data = load_dataset(args.dataset)
    logger.info(f"Loaded {args.dataset} dataset with {len(data)} entries")
    
    # Build graph
    graph = build_graph(data)
    logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Initialize model
    if args.model == "gcn":
        model = GCNModel(
            input_dim=graph.number_of_nodes(),
            embedding_dim=args.embedding_dim,
            num_layers=config.NUM_GNN_LAYERS
        )
    elif args.model == "gat":
        model = GATModel(
            input_dim=graph.number_of_nodes(),
            embedding_dim=args.embedding_dim,
            num_layers=config.NUM_GNN_LAYERS
        )
    elif args.model == "lightgcn":
        model = LightGCNModel(
            input_dim=graph.number_of_nodes(),
            embedding_dim=args.embedding_dim,
            num_layers=config.NUM_GNN_LAYERS
        )
    elif args.model == "graphsage":
        model = GraphSAGEModel(
            input_dim=graph.number_of_nodes(),
            embedding_dim=args.embedding_dim,
            num_layers=config.NUM_GNN_LAYERS
        )
    
    # Train model
    logger.info(f"Training {args.model} model for {args.epochs} epochs")
    model.train(
        graph=graph,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Save model
    model_path = os.path.join(config.MODELS_DIR, f"{args.model}_{args.dataset}.pt")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

def evaluate(args):
    """Evaluate the GNN model."""
    logger.info(f"Evaluating {args.model} model on {args.dataset} dataset")
    
    # Load dataset
    data = load_dataset(args.dataset)
    logger.info(f"Loaded {args.dataset} dataset with {len(data)} entries")
    
    # Build graph
    graph = build_graph(data)
    logger.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Load model
    model_path = os.path.join(config.MODELS_DIR, f"{args.model}_{args.dataset}.pt")
    if args.model == "gcn":
        model = GCNModel.load(model_path)
    elif args.model == "gat":
        model = GATModel.load(model_path)
    elif args.model == "lightgcn":
        model = LightGCNModel.load(model_path)
    elif args.model == "graphsage":
        model = GraphSAGEModel.load(model_path)
    
    # Evaluate model
    metrics = model.evaluate(graph)
    
    # Print metrics
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value:.4f}")

def serve():
    """Start the API server."""
    logger.info("Starting API server")
    
    # Import here to avoid circular imports
    from api.server import start_server
    start_server()

def main():
    """Main entry point."""
    args = parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "serve":
        serve()

if __name__ == "__main__":
    main() 