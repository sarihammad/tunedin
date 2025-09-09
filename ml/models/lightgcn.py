"""LightGCN implementation for music recommendation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class LightGCNConv(MessagePassing):
    """Light Graph Convolutional Network layer."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x, edge_index):
        """Forward pass."""
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Normalize adjacency matrix
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j, norm):
        """Message function."""
        return norm.view(-1, 1) * x_j


class LightGCN(nn.Module):
    """LightGCN model for collaborative filtering."""
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # LightGCN layers
        self.convs = nn.ModuleList([
            LightGCNConv(embedding_dim, embedding_dim)
            for _ in range(num_layers)
        ])
        
        # Layer weights for final aggregation
        self.layer_weights = nn.Parameter(torch.ones(num_layers + 1))
    
    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index, user_ids=None, item_ids=None):
        """Forward pass."""
        # Get initial embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Combine user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Store embeddings from each layer
        embeddings = [all_emb]
        
        # Apply LightGCN layers
        for conv in self.convs:
            all_emb = conv(all_emb, edge_index)
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embeddings.append(all_emb)
        
        # Weighted aggregation of all layers
        final_emb = torch.zeros_like(embeddings[0])
        for i, emb in enumerate(embeddings):
            final_emb += self.layer_weights[i] * emb
        
        # Normalize
        final_emb = F.normalize(final_emb, p=2, dim=1)
        
        # Split back into user and item embeddings
        user_emb_final = final_emb[:self.num_users]
        item_emb_final = final_emb[self.num_users:]
        
        if user_ids is not None and item_ids is not None:
            # Return specific user and item embeddings
            return user_emb_final[user_ids], item_emb_final[item_ids]
        
        return user_emb_final, item_emb_final
    
    def get_embeddings(self):
        """Get final user and item embeddings."""
        with torch.no_grad():
            # Create dummy edge index for forward pass
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.user_embedding.weight.device)
            user_emb, item_emb = self.forward(edge_index)
            return user_emb, item_emb


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    """Bayesian Personalized Ranking loss."""
    pos_scores = torch.sum(user_emb * pos_item_emb, dim=1)
    neg_scores = torch.sum(user_emb * neg_item_emb, dim=1)
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
    return loss


def create_bipartite_edge_index(user_ids, item_ids, num_users, num_items):
    """Create bipartite edge index for user-item interactions."""
    # Convert item IDs to global indices (add num_users offset)
    global_item_ids = item_ids + num_users
    
    # Create edge index
    edge_index = torch.stack([
        torch.cat([user_ids, global_item_ids]),
        torch.cat([global_item_ids, user_ids])
    ], dim=0)
    
    return edge_index


def compute_metrics(model, test_data, k=10):
    """Compute evaluation metrics (nDCG@k, HR@k)."""
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
        
        # Compute all user-item scores
        scores = torch.mm(user_emb, item_emb.t())
        
        ndcg_scores = []
        hr_scores = []
        
        for user_id, pos_items in test_data.items():
            if user_id >= scores.size(0):
                continue
                
            user_scores = scores[user_id]
            
            # Get top-k items
            _, top_k_indices = torch.topk(user_scores, k)
            top_k_items = set(top_k_indices.cpu().numpy())
            
            # Compute HR@k
            hr = len(top_k_items.intersection(set(pos_items))) / min(k, len(pos_items))
            hr_scores.append(hr)
            
            # Compute nDCG@k
            dcg = 0.0
            for i, item_id in enumerate(top_k_indices):
                if item_id.item() in pos_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(pos_items))))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcg_scores.append(ndcg)
        
        return {
            'ndcg@k': np.mean(ndcg_scores),
            'hr@k': np.mean(hr_scores)
        }

