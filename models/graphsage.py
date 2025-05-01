"""
GraphSAGE model for recommendation.
GraphSAGE is particularly good for inductive learning and handling new nodes.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from loguru import logger
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.evaluation import evaluate_recommendations

class GraphSAGEEncoder(torch.nn.Module):
    """
    GraphSAGE Encoder for node embeddings.
    """
    def __init__(self, hidden_channels, out_channels, num_layers=2, aggr='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.aggr = aggr
        
        # Create convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(-1, hidden_channels, aggr=aggr))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            
        Returns:
            torch.Tensor: Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        return x

class GraphSAGEDecoder(torch.nn.Module):
    """
    Decoder for link prediction.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
    
    def forward(self, z_src, z_dst):
        """
        Forward pass.
        
        Args:
            z_src (torch.Tensor): Source node embeddings
            z_dst (torch.Tensor): Destination node embeddings
            
        Returns:
            torch.Tensor: Predicted scores
        """
        # Concatenate user and item embeddings
        z = torch.cat([z_src, z_dst], dim=-1)
        z = self.lin1(z).relu()
        z = F.dropout(z, p=0.2, training=self.training)
        z = self.lin2(z)
        return z.sigmoid()

class GraphSAGEModel(torch.nn.Module):
    """
    GraphSAGE model for recommendation.
    """
    def __init__(self, input_dim, embedding_dim=128, num_layers=2, aggr='mean', user_feature_dim=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.aggr = aggr

        if user_feature_dim is None:
            raise ValueError("user_feature_dim must be provided for cold-start projection")
        self.user_feature_dim = user_feature_dim

        # Projection layer for user features
        self.user_feature_projection = nn.Sequential(
            nn.Linear(user_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

        # Optionally, projection for item features can be added here as well
        # self.item_feature_projection = nn.Sequential(
        #     nn.Linear(item_feature_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.LayerNorm(embedding_dim)
        # )
        
        # Create base GNN model
        self.encoder = GraphSAGEEncoder(
            hidden_channels=embedding_dim,
            out_channels=embedding_dim,
            num_layers=num_layers,
            aggr=aggr
        )
        
        # Create decoder for link prediction
        self.decoder = GraphSAGEDecoder(hidden_channels=embedding_dim)
        
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
        
        logger.info(f"Training GraphSAGE model on {device} for {epochs} epochs")
        
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
    
    def inductive_recommend(self, user_features, item_features, top_k=10):
        """
        Generate recommendations for new users or items not seen during training.
        This is the key advantage of GraphSAGE - inductive learning capability.
        
        Args:
            user_features (torch.Tensor): Features of new users
            item_features (torch.Tensor): Features of all items
            top_k (int): Number of recommendations to generate
            
        Returns:
            list: List of lists of recommended item indices
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making recommendations")
        
        # Create a dummy graph for the new users
        dummy_graph = HeteroData()
        
        # Project user features to embedding space
        projected_user_features = self.user_feature_projection(user_features)
        dummy_graph[config.USER_NODE_TYPE].x = projected_user_features
        # Optionally, project item features if needed:
        # projected_item_features = self.item_feature_projection(item_features)
        dummy_graph[config.SONG_NODE_TYPE].x = item_features
        
        # Add dummy edges (connect each user to a dummy item)
        num_users = user_features.size(0)
        num_items = item_features.size(0)
        
        # Create dummy edge index
        dummy_edge_index = torch.zeros((2, num_users), dtype=torch.long)
        dummy_edge_index[0] = torch.arange(num_users)
        dummy_edge_index[1] = torch.zeros(num_users, dtype=torch.long)
        
        dummy_graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index = dummy_edge_index
        
        # Forward pass to get embeddings
        with torch.no_grad():
            embeddings = self(dummy_graph)
        
        # Get user embeddings
        new_user_emb = embeddings[config.USER_NODE_TYPE]
        
        # Calculate similarity scores with all items
        scores = torch.matmul(new_user_emb, self.item_embedding.t())
        
        # Convert to numpy for easier manipulation
        scores_np = scores.detach().cpu().numpy()
        
        # Get top-k items for each user
        recommendations = []
        for user_scores in scores_np:
            top_items = np.argsort(-user_scores)[:top_k]
            recommendations.append(top_items.tolist())
        
        return recommendations
    
    def save(self, path):
        """
        Save the model to a file.
        
        Args:
            path (str): Path to save the model to
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'embedding_dim': self.embedding_dim,
            'num_layers': self.num_layers,
            'aggr': self.aggr,
            'is_trained': self.is_trained,
            'user_embedding': self.user_embedding,
            'item_embedding': self.item_embedding,
            # Save user_feature_dim for loading
            'user_feature_dim': getattr(self, "user_feature_dim", 20)  # fallback to 20 if not present
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a model from a file.
        
        Args:
            path (str): Path to load the model from
            
        Returns:
            GraphSAGEModel: Loaded model
        """
        checkpoint = torch.load(path)
        # Use saved user_feature_dim or fallback to 20 (document as assumption)
        user_feature_dim = checkpoint.get('user_feature_dim', 20)  # Assumed default if not available
        model = cls(
            input_dim=None,  # Not needed when loading
            embedding_dim=checkpoint['embedding_dim'],
            num_layers=checkpoint['num_layers'],
            aggr=checkpoint.get('aggr', 'mean'),
            user_feature_dim=user_feature_dim
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.is_trained = checkpoint['is_trained']
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        
        logger.info(f"Model loaded from {path}")
        
        return model 