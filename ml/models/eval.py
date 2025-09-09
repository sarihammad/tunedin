"""Evaluation script for LightGCN model."""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .lightgcn import LightGCN, compute_metrics


def load_model_and_data():
    """Load trained model and test data."""
    models_dir = Path("models")
    
    # Load mappings
    with open(models_dir / "user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    
    with open(models_dir / "item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)
    
    # Load embeddings
    user_emb = np.load(models_dir / "user_emb.npy")
    item_emb = np.load(models_dir / "item_emb.npy")
    
    # Load test data
    data_dir = Path("data/processed")
    user_events = pd.read_parquet(data_dir / "user_events.parquet")
    
    # Map interactions
    user_events['user_idx'] = user_events['user_id'].map(user_mapping)
    user_events['item_idx'] = user_events['artist_id'].map(item_mapping)
    user_events = user_events.dropna(subset=['user_idx', 'item_idx'])
    
    # Split into train/test
    train_data, test_data = train_test_split(
        user_events, test_size=0.2, random_state=42
    )
    
    # Create test data structure
    test_dict = {}
    for _, row in test_data.iterrows():
        user_id = int(row['user_idx'])
        item_id = int(row['item_idx'])
        if user_id not in test_dict:
            test_dict[user_id] = []
        test_dict[user_id].append(item_id)
    
    return user_emb, item_emb, test_dict, len(user_mapping), len(item_mapping)


def evaluate_model(k_values=[5, 10, 20]):
    """Evaluate the trained model."""
    print("Loading model and data...")
    
    user_emb, item_emb, test_data, num_users, num_items = load_model_and_data()
    
    # Convert to tensors
    user_emb = torch.tensor(user_emb, dtype=torch.float32)
    item_emb = torch.tensor(item_emb, dtype=torch.float32)
    
    print(f"Loaded embeddings: users={user_emb.shape}, items={item_emb.shape}")
    print(f"Test users: {len(test_data)}")
    
    # Compute metrics for different k values
    results = {}
    
    for k in k_values:
        print(f"\nEvaluating with k={k}...")
        
        ndcg_scores = []
        hr_scores = []
        
        # Compute all user-item scores
        scores = torch.mm(user_emb, item_emb.t())
        
        for user_id, pos_items in tqdm(test_data.items(), desc=f"Evaluating k={k}"):
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
        
        results[k] = {
            'ndcg': np.mean(ndcg_scores),
            'hr': np.mean(hr_scores),
            'ndcg_std': np.std(ndcg_scores),
            'hr_std': np.std(hr_scores)
        }
        
        print(f"k={k}: nDCG@{k}={results[k]['ndcg']:.4f}±{results[k]['ndcg_std']:.4f}, "
              f"HR@{k}={results[k]['hr']:.4f}±{results[k]['hr_std']:.4f}")
    
    return results


def analyze_recommendations():
    """Analyze recommendation quality and diversity."""
    print("\nAnalyzing recommendation quality...")
    
    user_emb, item_emb, test_data, num_users, num_items = load_model_and_data()
    
    # Convert to tensors
    user_emb = torch.tensor(user_emb, dtype=torch.float32)
    item_emb = torch.tensor(item_emb, dtype=torch.float32)
    
    # Compute all scores
    scores = torch.mm(user_emb, item_emb.t())
    
    # Get top-10 recommendations for each user
    _, top_indices = torch.topk(scores, 10, dim=1)
    
    # Analyze diversity (intra-list diversity)
    diversity_scores = []
    for user_id in range(min(100, scores.size(0))):  # Sample 100 users
        user_top_items = top_indices[user_id].cpu().numpy()
        
        # Compute pairwise cosine similarity
        item_vectors = item_emb[user_top_items]
        similarities = torch.mm(item_vectors, item_vectors.t())
        
        # Average pairwise similarity (lower is more diverse)
        avg_similarity = similarities.mean().item()
        diversity = 1 - avg_similarity  # Convert to diversity score
        diversity_scores.append(diversity)
    
    print(f"Average diversity score: {np.mean(diversity_scores):.4f}")
    
    # Analyze popularity bias
    # Load item popularity (simplified)
    data_dir = Path("data/processed")
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    
    # Count interactions per item
    user_events = pd.read_parquet(data_dir / "user_events.parquet")
    item_popularity = user_events.groupby('artist_id').size()
    
    # Map to item indices
    with open("models/item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)
    
    popularity_scores = []
    for user_id in range(min(100, scores.size(0))):
        user_top_items = top_indices[user_id].cpu().numpy()
        
        # Get popularity of recommended items
        user_popularity = []
        for item_idx in user_top_items:
            # Find original item ID
            for orig_id, mapped_idx in item_mapping.items():
                if mapped_idx == item_idx:
                    popularity = item_popularity.get(orig_id, 0)
                    user_popularity.append(popularity)
                    break
        
        if user_popularity:
            popularity_scores.append(np.mean(user_popularity))
    
    print(f"Average popularity of recommendations: {np.mean(popularity_scores):.2f}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate LightGCN model')
    parser.add_argument('--k', type=int, nargs='+', default=[5, 10, 20], 
                       help='K values for evaluation')
    parser.add_argument('--analyze', action='store_true', 
                       help='Perform detailed analysis')
    
    args = parser.parse_args()
    
    print("TunedIn Model Evaluation")
    print("=" * 40)
    
    # Evaluate model
    results = evaluate_model(args.k)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 40)
    for k, metrics in results.items():
        print(f"k={k:2d}: nDCG@{k}={metrics['ndcg']:.4f}, HR@{k}={metrics['hr']:.4f}")
    
    # Detailed analysis
    if args.analyze:
        analyze_recommendations()
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

