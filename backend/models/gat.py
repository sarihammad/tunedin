"""
Graph Attention Network (GAT) model for recommendation.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.data import HeteroData
from loguru import logger
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import backend.config as config
from utils.evaluation import evaluate_recommendations

class GATEncoder(torch.nn.Module):
    """
    GAT Encoder for node embeddings.
    """
    def __init__(self, hidden_channels, out_channels, heads=4):
        super().__init__()
        self.conv1 = GATConv(-1, hidden_channels, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GATDecoder(torch.nn.Module):
    """
    Decoder for link prediction.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, z_src, z_dst):
        # Concatenate user and item embeddings
        z = torch.cat([z_src, z_dst], dim=-1)
        z = self.lin1(z).relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.lin2(z)
        return z.sigmoid()

class GATModel(torch.nn.Module):
    """
    GAT model for recommendation.
    """
    def __init__(self, input_dim, embedding_dim=128, num_layers=2, heads=4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.heads = heads
        
        # Create base GNN model
        self.encoder = GATEncoder(
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            heads=heads
        )
        
        # Create decoder for link prediction
        self.decoder = GATDecoder(hidden_channels=embedding_dim)
        
        # Initialize embeddings
        self.user_embedding = None
        self.item_embedding = None
        
        # Training state
        self.is_trained = False
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data (HeteroData): Heterogeneous graph data
            
        Returns:
            dict: Dictionary of node embeddings
        """
        # Apply GNN to get node embeddings
        if isinstance(data, HeteroData):
            # For heterogeneous graphs, convert the base model
            encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
            
            # Get node embeddings for each node type
            embeddings = encoder(data.x_dict, data.edge_index_dict)
            
            # Store user and item embeddings
            self.user_embedding = embeddings[config.USER_NODE_TYPE]
            self.item_embedding = embeddings[config.SONG_NODE_TYPE]
            
            return embeddings
        else:
            # For homogeneous graphs
            embeddings = self.encoder(data.x, data.edge_index)
            
            # Split embeddings by node type
            user_mask = data.node_type == 0  # Assuming user nodes have type 0
            item_mask = data.node_type == 1  # Assuming item nodes have type 1
            
            self.user_embedding = embeddings[user_mask]
            self.item_embedding = embeddings[item_mask]
            
            return {
                config.USER_NODE_TYPE: self.user_embedding,
                config.SONG_NODE_TYPE: self.item_embedding
            }
    
    def predict(self, user_indices, item_indices):
        """
        Predict scores for user-item pairs.
        
        Args:
            user_indices (torch.Tensor): User indices
            item_indices (torch.Tensor): Item indices
            
        Returns:
            torch.Tensor: Predicted scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Get embeddings for the specified users and items
        user_emb = self.user_embedding[user_indices]
        item_emb = self.item_embedding[item_indices]
        
        # Predict scores
        return self.decoder(user_emb, item_emb)
    
    def recommend(self, user_indices, top_k=10, exclude_interacted=True, interacted_items=None):
        """
        Generate recommendations for users.
        
        Args:
            user_indices (torch.Tensor): User indices
            top_k (int): Number of recommendations to generate
            exclude_interacted (bool): Whether to exclude items the user has interacted with
            interacted_items (dict): Dictionary mapping user indices to lists of interacted item indices
            
        Returns:
            list: List of lists of recommended item indices
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making recommendations")
        
        # Get user embeddings
        user_emb = self.user_embedding[user_indices]
        
        # Calculate similarity scores with all items
        scores = torch.matmul(user_emb, self.item_embedding.t())
        
        # Convert to numpy for easier manipulation
        scores_np = scores.detach().cpu().numpy()
        
        recommendations = []
        
        for i, user_idx in enumerate(user_indices):
            user_scores = scores_np[i]
            
            # Exclude interacted items if requested
            if exclude_interacted and interacted_items is not None and user_idx in interacted_items:
                user_interacted = interacted_items[user_idx]
                user_scores[user_interacted] = -np.inf
            
            # Get top-k items
            top_items = np.argsort(-user_scores)[:top_k]
            recommendations.append(top_items.tolist())
        
        return recommendations
    
    def train(self, graph, epochs=100, batch_size=1024, learning_rate=0.001, device=None):
        """
        Train the model.
        
        Args:
            graph (HeteroData): Graph data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            learning_rate (float): Learning rate
            device (str): Device to train on ('cuda' or 'cpu')
            
        Returns:
            dict: Training history
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Training GAT model on {device} for {epochs} epochs")
        
        # Move model and data to device
        self.to(device)
        graph = graph.to(device)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training step
            self.train()
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self(graph)
            
            # Sample positive and negative edges
            pos_edge_index = graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index
            
            # Create positive samples
            pos_user_indices = pos_edge_index[0]
            pos_item_indices = pos_edge_index[1]
            
            # Create negative samples
            neg_user_indices = pos_user_indices
            neg_item_indices = torch.randint(
                0, graph[config.SONG_NODE_TYPE].num_nodes, 
                (pos_user_indices.size(0),), 
                device=device
            )
            
            # Predict scores
            pos_scores = self.predict(pos_user_indices, pos_item_indices)
            neg_scores = self.predict(neg_user_indices, neg_item_indices)
            
            # Compute loss
            loss = F.binary_cross_entropy(
                pos_scores, 
                torch.ones_like(pos_scores)
            ) + F.binary_cross_entropy(
                neg_scores, 
                torch.zeros_like(neg_scores)
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Log progress
            history['loss'].append(loss.item())
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        # Mark as trained
        self.is_trained = True
        
        # Move model back to CPU for inference
        self.to('cpu')
        
        return history
    
    def evaluate(self, graph, k=10):
        """
        Evaluate the model.
        
        Args:
            graph (HeteroData): Graph data
            k (int): Number of recommendations to consider
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        # Set model to evaluation mode
        self.eval()
        
        # Get user-item interactions
        edge_index = graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index
        
        # Create a dictionary of user interactions
        user_interactions = {}
        for i in range(edge_index.size(1)):
            user_idx = edge_index[0, i].item()
            item_idx = edge_index[1, i].item()
            
            if user_idx not in user_interactions:
                user_interactions[user_idx] = []
            
            user_interactions[user_idx].append(item_idx)
        
        # Split interactions into train and test
        train_interactions = {}
        test_interactions = {}
        
        for user_idx, items in user_interactions.items():
            # Use 80% for training, 20% for testing
            split_idx = int(len(items) * 0.8)
            train_interactions[user_idx] = items[:split_idx]
            test_interactions[user_idx] = items[split_idx:]
        
        # Generate recommendations
        user_indices = torch.tensor(list(user_interactions.keys()))
        recommendations = self.recommend(
            user_indices, 
            top_k=k, 
            exclude_interacted=True, 
            interacted_items=train_interactions
        )
        
        # Prepare data for evaluation
        y_true = [test_interactions[user_idx.item()] for user_idx in user_indices]
        y_pred = recommendations
        
        # Evaluate recommendations
        metrics = evaluate_recommendations(y_true, y_pred, k=k)
        
        return metrics
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'heads': self.heads,
            'is_trained': self.is_trained,
            'user_embedding': self.user_embedding,
            'item_embedding': self.item_embedding
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a model from a file.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            GATModel: Loaded model
        """
        checkpoint = torch.load(path)
        
        # Create model
        model = cls(
            input_dim=None,  # Not needed when loading
            embedding_dim=checkpoint['embedding_dim'],
            heads=checkpoint.get('heads', 4)
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.is_trained = checkpoint['is_trained']
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        
        logger.info(f"Model loaded from {path}")
        
        return model 