"""Training script for LightGCN model."""

import os
import pickle
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from .lightgcn import LightGCN, bpr_loss, create_bipartite_edge_index, compute_metrics


class MusicDataset(Dataset):
    """Dataset for music recommendation training."""
    
    def __init__(self, interactions, num_users, num_items):
        self.interactions = interactions
        self.num_users = num_users
        self.num_items = num_items
    
    def __len__(self):
        return len(self.interactions)
    
    def __getitem__(self, idx):
        user_id, item_id, play_count = self.interactions[idx]
        return torch.tensor(user_id, dtype=torch.long), torch.tensor(item_id, dtype=torch.long), play_count


def load_data():
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    
    # Load processed data
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print("Processed data not found. Please run data processing first.")
        return None
    
    # Load interactions
    user_events = pd.read_parquet(data_dir / "user_events.parquet")
    users = pd.read_parquet(data_dir / "users.parquet")
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    
    print(f"Loaded {len(user_events)} interactions")
    print(f"Users: {len(users)}, Tracks: {len(tracks)}")
    
    # Create user and item mappings
    user_mapping = {uid: idx for idx, uid in enumerate(users['user_id'].unique())}
    item_mapping = {iid: idx for idx, iid in enumerate(tracks['track_id'].unique())}
    
    # Map interactions
    user_events['user_idx'] = user_events['user_id'].map(user_mapping)
    user_events['item_idx'] = user_events['artist_id'].map(item_mapping)
    
    # Remove missing mappings
    user_events = user_events.dropna(subset=['user_idx', 'item_idx'])
    user_events['user_idx'] = user_events['user_idx'].astype(int)
    user_events['item_idx'] = user_events['item_idx'].astype(int)
    
    print(f"After mapping: {len(user_events)} interactions")
    
    return {
        'interactions': user_events[['user_idx', 'item_idx', 'play_count']].values,
        'num_users': len(user_mapping),
        'num_items': len(item_mapping),
        'user_mapping': user_mapping,
        'item_mapping': item_mapping,
        'users': users,
        'tracks': tracks
    }


def create_negative_samples(interactions, num_users, num_items, num_negatives=1):
    """Create negative samples for BPR loss."""
    print("Creating negative samples...")
    
    # Get all positive interactions
    positive_interactions = set()
    for user_id, item_id, _ in interactions:
        positive_interactions.add((user_id, item_id))
    
    # Create negative samples
    negative_samples = []
    for user_id, item_id, play_count in interactions:
        for _ in range(num_negatives):
            # Sample negative item
            neg_item_id = np.random.randint(0, num_items)
            while (user_id, neg_item_id) in positive_interactions:
                neg_item_id = np.random.randint(0, num_items)
            
            negative_samples.append((user_id, item_id, neg_item_id, play_count))
    
    return np.array(negative_samples)


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        user_ids, pos_item_ids, neg_item_ids, _ = batch
        user_ids = user_ids.to(device)
        pos_item_ids = pos_item_ids.to(device)
        neg_item_ids = neg_item_ids.to(device)
        
        # Get embeddings
        user_emb, pos_item_emb = model(None, user_ids, pos_item_ids)
        _, neg_item_emb = model(None, user_ids, neg_item_ids)
        
        # Compute BPR loss
        loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, test_data, device, k=10):
    """Evaluate the model."""
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


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train LightGCN model')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data = load_data()
    if data is None:
        return
    
    # Split data
    interactions = data['interactions']
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    # Create negative samples
    train_negatives = create_negative_samples(
        train_interactions, 
        data['num_users'], 
        data['num_items']
    )
    
    # Create test data (user -> positive items)
    test_data = {}
    for user_id, item_id, _ in test_interactions:
        if user_id not in test_data:
            test_data[user_id] = []
        test_data[user_id].append(item_id)
    
    # Create datasets
    train_dataset = MusicDataset(train_negatives, data['num_users'], data['num_items'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create model
    model = LightGCN(
        num_users=data['num_users'],
        num_items=data['num_items'],
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_metrics = []
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        if epoch % 10 == 0:
            metrics = evaluate(model, test_data, device)
            val_metrics.append(metrics)
            print(f"Epoch {epoch}: Loss={train_loss:.4f}, "
                  f"nDCG@10={metrics['ndcg@k']:.4f}, HR@10={metrics['hr@k']:.4f}")
    
    # Save model and embeddings
    print("Saving model and embeddings...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), models_dir / "lightgcn_model.pt")
    
    # Save embeddings
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
        np.save(models_dir / "user_emb.npy", user_emb.cpu().numpy())
        np.save(models_dir / "item_emb.npy", item_emb.cpu().numpy())
    
    # Save mappings
    with open(models_dir / "user_mapping.pkl", 'wb') as f:
        pickle.dump(data['user_mapping'], f)
    
    with open(models_dir / "item_mapping.pkl", 'wb') as f:
        pickle.dump(data['item_mapping'], f)
    
    # Save item IDs for FAISS index
    item_ids = np.array(list(data['item_mapping'].keys()))
    np.save(models_dir / "item_ids.npy", item_ids)
    
    print("Training completed!")
    print(f"Final metrics: nDCG@10={val_metrics[-1]['ndcg@k']:.4f}, "
          f"HR@10={val_metrics[-1]['hr@k']:.4f}")


if __name__ == "__main__":
    main()

