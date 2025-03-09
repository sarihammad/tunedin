"""
Graph building utilities for the TunedIn project.
"""
import os
import networkx as nx
import pandas as pd
import numpy as np
from loguru import logger
import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def build_graph(dataset):
    """
    Build a graph from the dataset.
    
    Args:
        dataset (dict): Dictionary containing songs and interactions dataframes
        
    Returns:
        nx.Graph or torch_geometric.data.Data: Graph representation of the dataset
    """
    logger.info("Building graph from dataset")
    
    # Extract dataframes from dataset
    songs_df = dataset["songs"]
    interactions_df = dataset["interactions"]
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add song nodes
    logger.info(f"Adding {len(songs_df)} song nodes to the graph")
    for _, song in songs_df.iterrows():
        # Add song node
        G.add_node(
            song["song_id"],
            node_type=config.SONG_NODE_TYPE,
            name=song["song_name"],
            features={feature: song[feature] for feature in config.AUDIO_FEATURES if feature in song}
        )
        
        # Add artist node if it doesn't exist
        if not G.has_node(song["artist_id"]):
            G.add_node(
                song["artist_id"],
                node_type=config.ARTIST_NODE_TYPE,
                name=song["artist_name"]
            )
        
        # Add album node if it doesn't exist
        if not G.has_node(song["album_id"]):
            G.add_node(
                song["album_id"],
                node_type=config.ALBUM_NODE_TYPE,
                name=song["album_name"]
            )
        
        # Add genre node if it doesn't exist
        genre = song["genre"]
        if not G.has_node(f"genre_{genre}"):
            G.add_node(
                f"genre_{genre}",
                node_type=config.GENRE_NODE_TYPE,
                name=genre
            )
        
        # Add edges
        G.add_edge(song["song_id"], song["artist_id"], edge_type=config.PERFORMED_BY_EDGE)
        G.add_edge(song["song_id"], song["album_id"], edge_type=config.PART_OF_EDGE)
        G.add_edge(song["song_id"], f"genre_{genre}", edge_type=config.BELONGS_TO_EDGE)
    
    # Add user nodes and interaction edges
    logger.info(f"Adding user nodes and interaction edges to the graph")
    for _, interaction in interactions_df.iterrows():
        user_id = interaction["user_id"]
        song_id = interaction["song_id"]
        interaction_type = interaction["interaction_type"]
        interaction_count = interaction["interaction_count"]
        
        # Add user node if it doesn't exist
        if not G.has_node(user_id):
            G.add_node(
                user_id,
                node_type=config.USER_NODE_TYPE
            )
        
        # Add edge if the song exists in the graph
        if G.has_node(song_id):
            G.add_edge(
                user_id,
                song_id,
                edge_type=interaction_type,
                weight=interaction_count
            )
    
    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Convert to PyTorch Geometric format for GNN models
    pyg_graph = convert_to_pytorch_geometric(G)
    
    return pyg_graph

def convert_to_pytorch_geometric(G):
    """
    Convert a NetworkX graph to a PyTorch Geometric graph.
    
    Args:
        G (nx.Graph): NetworkX graph
        
    Returns:
        torch_geometric.data.Data: PyTorch Geometric graph
    """
    logger.info("Converting NetworkX graph to PyTorch Geometric format")
    
    # Create a heterogeneous graph
    data = HeteroData()
    
    # Get all node types
    node_types = set(nx.get_node_attributes(G, "node_type").values())
    
    # Create node type mappings
    node_type_to_idx = {node_type: {} for node_type in node_types}
    
    # Add nodes for each node type
    for node_type in node_types:
        # Get nodes of this type
        nodes = [node for node, attrs in G.nodes(data=True) if attrs.get("node_type") == node_type]
        
        # Create mapping from original node ID to index
        for i, node in enumerate(nodes):
            node_type_to_idx[node_type][node] = i
        
        # Add node features
        if node_type == config.SONG_NODE_TYPE:
            # For songs, use audio features
            features = []
            for node in nodes:
                node_features = G.nodes[node].get("features", {})
                # Convert features to a list in a consistent order
                feature_vector = [
                    node_features.get(feature, 0.0) 
                    for feature in config.AUDIO_FEATURES
                ]
                features.append(feature_vector)
            
            # Convert to tensor
            data[node_type].x = torch.tensor(features, dtype=torch.float)
        else:
            # For other node types, use one-hot encoding
            data[node_type].x = torch.eye(len(nodes), dtype=torch.float)
        
        # Store the number of nodes
        data[node_type].num_nodes = len(nodes)
    
    # Add edges for each edge type
    edge_types = set(nx.get_edge_attributes(G, "edge_type").values())
    
    for edge_type in edge_types:
        # Get edges of this type
        edges = [
            (u, v) for u, v, attrs in G.edges(data=True) 
            if attrs.get("edge_type") == edge_type
        ]
        
        if not edges:
            continue
        
        # Determine source and target node types
        src_type = G.nodes[edges[0][0]]["node_type"]
        dst_type = G.nodes[edges[0][1]]["node_type"]
        
        # Create edge index tensor
        edge_index = []
        edge_weights = []
        
        for u, v in edges:
            # Map node IDs to indices
            u_idx = node_type_to_idx[src_type][u]
            v_idx = node_type_to_idx[dst_type][v]
            
            edge_index.append([u_idx, v_idx])
            
            # Get edge weight if available
            weight = G.edges[u, v].get("weight", 1.0)
            edge_weights.append(weight)
        
        # Convert to tensor
        data[src_type, edge_type, dst_type].edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        data[src_type, edge_type, dst_type].edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
    
    # Apply transforms
    data = ToUndirected()(data)
    
    logger.info(f"Converted to PyTorch Geometric format with {len(node_types)} node types and {len(edge_types)} edge types")
    
    return data

def create_user_item_matrix(interactions_df, user_col="user_id", item_col="song_id", rating_col="interaction_count"):
    """
    Create a user-item interaction matrix from a dataframe of interactions.
    
    Args:
        interactions_df (pd.DataFrame): Dataframe of user-item interactions
        user_col (str): Name of the user column
        item_col (str): Name of the item column
        rating_col (str): Name of the rating column
        
    Returns:
        scipy.sparse.csr_matrix: Sparse user-item matrix
    """
    # Create user and item mappings
    users = interactions_df[user_col].unique()
    items = interactions_df[item_col].unique()
    
    user_to_idx = {user: i for i, user in enumerate(users)}
    item_to_idx = {item: i for i, item in enumerate(items)}
    
    # Create matrix
    user_indices = interactions_df[user_col].map(user_to_idx).values
    item_indices = interactions_df[item_col].map(item_to_idx).values
    ratings = interactions_df[rating_col].values
    
    # Create sparse matrix
    from scipy.sparse import csr_matrix
    matrix = csr_matrix((ratings, (user_indices, item_indices)), shape=(len(users), len(items)))
    
    return matrix, user_to_idx, item_to_idx

def split_train_val_test(interactions_df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, by_user=True):
    """
    Split interactions into train, validation, and test sets.
    
    Args:
        interactions_df (pd.DataFrame): Dataframe of user-item interactions
        train_ratio (float): Ratio of interactions to use for training
        val_ratio (float): Ratio of interactions to use for validation
        test_ratio (float): Ratio of interactions to use for testing
        by_user (bool): Whether to split by user (each user has interactions in all sets)
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    if by_user:
        # Split interactions for each user
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for user, user_interactions in interactions_df.groupby("user_id"):
            # Shuffle interactions
            user_interactions = user_interactions.sample(frac=1, random_state=42)
            
            # Calculate split indices
            n = len(user_interactions)
            train_idx = int(n * train_ratio)
            val_idx = int(n * (train_ratio + val_ratio))
            
            # Split
            train_dfs.append(user_interactions.iloc[:train_idx])
            val_dfs.append(user_interactions.iloc[train_idx:val_idx])
            test_dfs.append(user_interactions.iloc[val_idx:])
        
        # Combine
        train_df = pd.concat(train_dfs)
        val_df = pd.concat(val_dfs)
        test_df = pd.concat(test_dfs)
    else:
        # Split all interactions
        interactions_df = interactions_df.sample(frac=1, random_state=42)
        
        # Calculate split indices
        n = len(interactions_df)
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        # Split
        train_df = interactions_df.iloc[:train_idx]
        val_df = interactions_df.iloc[train_idx:val_idx]
        test_df = interactions_df.iloc[val_idx:]
    
    return train_df, val_df, test_df 