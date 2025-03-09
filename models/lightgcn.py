"""
LightGCN model for recommendation.
LightGCN is a simplified GCN specifically designed for recommendation systems.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_sparse import SparseTensor
from loguru import logger
import numpy as np
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.evaluation import evaluate_recommendations

class LightGCNEncoder(torch.nn.Module):
    """
    LightGCN Encoder for node embeddings.
    """
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize weights
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
    
    def forward(self, edge_index):
        """
        Forward pass.
        
        Args:
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Create adjacency matrix
        adj = self._create_adj_matrix(edge_index)
        
        # Storage for embeddings from each layer
        user_embs = [user_emb]
        item_embs = [item_emb]
        
        # Message passing
        emb = torch.cat([user_emb, item_emb], dim=0)
        
        for _ in range(self.num_layers):
            # Light graph convolution: only neighborhood aggregation, no feature transformation
            emb = torch.sparse.mm(adj, emb)
            
            # Split user and item embeddings
            user_emb, item_emb = torch.split(emb, [self.num_users, self.num_items])
            
            # Store embeddings
            user_embs.append(user_emb)
            item_embs.append(item_emb)
        
        # Final embeddings are the mean of all layers (including the initial embeddings)
        final_user_emb = torch.stack(user_embs, dim=0).mean(dim=0)
        final_item_emb = torch.stack(item_embs, dim=0).mean(dim=0)
        
        return final_user_emb, final_item_emb
    
    def _create_adj_matrix(self, edge_index):
        """
        Create a normalized adjacency matrix.
        
        Args:
            edge_index (torch.Tensor): Edge indices of shape [2, num_edges]
            
        Returns:
            torch.sparse.FloatTensor: Normalized adjacency matrix
        """
        # Create adjacency matrix
        adj = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            sparse_sizes=(self.num_users + self.num_items, self.num_users + self.num_items)
        )
        
        # Make it symmetric
        adj = adj.to_symmetric()
        
        # Compute degree
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize adjacency matrix
        adj = SparseTensor(
            row=adj.storage.row(),
            col=adj.storage.col(),
            value=deg_inv_sqrt[adj.storage.row()] * adj.storage.value() * deg_inv_sqrt[adj.storage.col()],
            sparse_sizes=adj.sparse_sizes()
        )
        
        return adj.to_torch_sparse_coo_tensor()

class LightGCNModel(torch.nn.Module):
    """
    LightGCN model for recommendation.
    """
    def __init__(self, input_dim, embedding_dim=64, num_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # These will be set when preparing the model for training
        self.num_users = None
        self.num_items = None
        self.encoder = None
        
        # Initialize embeddings
        self.user_embedding = None
        self.item_embedding = None
        
        # Training state
        self.is_trained = False
    
    def _prepare_model(self, graph):
        """
        Prepare the model for training.
        
        Args:
            graph (HeteroData): Heterogeneous graph data
        """
        # Get number of users and items
        self.num_users = graph[config.USER_NODE_TYPE].num_nodes
        self.num_items = graph[config.SONG_NODE_TYPE].num_nodes
        
        # Create encoder
        self.encoder = LightGCNEncoder(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers
        )
    
    def forward(self, graph):
        """
        Forward pass.
        
        Args:
            graph (HeteroData): Heterogeneous graph data
            
        Returns:
            tuple: (user_embeddings, item_embeddings)
        """
        # Prepare model if not already prepared
        if self.encoder is None:
            self._prepare_model(graph)
        
        # Get user-item interactions
        edge_index = self._get_edge_index(graph)
        
        # Get embeddings
        self.user_embedding, self.item_embedding = self.encoder(edge_index)
        
        return self.user_embedding, self.item_embedding
    
    def _get_edge_index(self, graph):
        """
        Get edge index from graph.
        
        Args:
            graph (HeteroData): Heterogeneous graph data
            
        Returns:
            torch.Tensor: Edge index
        """
        # Get user-item interactions
        edge_index = graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index.clone()
        
        # Shift item indices
        edge_index[1] += self.num_users
        
        return edge_index
    
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
        
        # Compute dot product
        scores = torch.sum(user_emb * item_emb, dim=1)
        
        # Apply sigmoid to get scores in [0, 1]
        return torch.sigmoid(scores)
    
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
        
        logger.info(f"Training LightGCN model on {device} for {epochs} epochs")
        
        # Prepare model if not already prepared
        if self.encoder is None:
            self._prepare_model(graph)
        
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
            user_emb, item_emb = self(graph)
            
            # Sample positive and negative edges
            pos_edge_index = self._get_edge_index(graph)
            
            # Create positive samples
            pos_user_indices = pos_edge_index[0]
            pos_item_indices = pos_edge_index[1] - self.num_users  # Shift back to original indices
            
            # Create negative samples (random items for each user)
            neg_user_indices = pos_user_indices
            neg_item_indices = torch.randint(
                0, self.num_items, 
                (pos_user_indices.size(0),), 
                device=device
            )
            
            # Get embeddings for positive and negative samples
            pos_user_emb = user_emb[pos_user_indices]
            pos_item_emb = item_emb[pos_item_indices]
            neg_user_emb = user_emb[neg_user_indices]
            neg_item_emb = item_emb[neg_item_indices]
            
            # Compute scores
            pos_scores = torch.sum(pos_user_emb * pos_item_emb, dim=1)
            neg_scores = torch.sum(neg_user_emb * neg_item_emb, dim=1)
            
            # BPR loss
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
            
            # L2 regularization
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)
            
            loss += 1e-5 * l2_reg
            
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
            'num_layers': self.num_layers,
            'num_users': self.num_users,
            'num_items': self.num_items,
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
            LightGCNModel: Loaded model
        """
        checkpoint = torch.load(path)
        
        # Create model
        model = cls(
            input_dim=None,  # Not needed when loading
            embedding_dim=checkpoint['embedding_dim'],
            num_layers=checkpoint['num_layers']
        )
        
        # Set model attributes
        model.num_users = checkpoint['num_users']
        model.num_items = checkpoint['num_items']
        
        # Create encoder
        model.encoder = LightGCNEncoder(
            num_users=model.num_users,
            num_items=model.num_items,
            embedding_dim=model.embedding_dim,
            num_layers=model.num_layers
        )
        
        # Load state
        model.load_state_dict(checkpoint['model_state_dict'])
        model.is_trained = checkpoint['is_trained']
        model.user_embedding = checkpoint['user_embedding']
        model.item_embedding = checkpoint['item_embedding']
        
        logger.info(f"Model loaded from {path}")
        
        return model 