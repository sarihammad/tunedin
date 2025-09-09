"""Export embeddings for serving."""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from .lightgcn import LightGCN


def load_model_and_embeddings():
    """Load trained model and extract embeddings."""
    models_dir = Path("models")
    
    # Load mappings
    with open(models_dir / "user_mapping.pkl", 'rb') as f:
        user_mapping = pickle.load(f)
    
    with open(models_dir / "item_mapping.pkl", 'rb') as f:
        item_mapping = pickle.load(f)
    
    # Load model
    model = LightGCN(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        embedding_dim=64,  # Should match training
        num_layers=3
    )
    
    # Load model weights
    model.load_state_dict(torch.load(models_dir / "lightgcn_model.pt", map_location='cpu'))
    model.eval()
    
    # Extract embeddings
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings()
    
    return user_emb.numpy(), item_emb.numpy(), user_mapping, item_mapping


def export_embeddings():
    """Export embeddings in the format expected by the serving system."""
    print("Loading model and extracting embeddings...")
    
    user_emb, item_emb, user_mapping, item_mapping = load_model_and_embeddings()
    
    print(f"User embeddings shape: {user_emb.shape}")
    print(f"Item embeddings shape: {item_emb.shape}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    print("Saving embeddings...")
    np.save(models_dir / "user_emb.npy", user_emb)
    np.save(models_dir / "item_emb.npy", item_emb)
    
    # Save item IDs in the order they appear in the embeddings
    # This is important for FAISS index alignment
    item_ids = np.array(list(item_mapping.keys()))
    np.save(models_dir / "item_ids.npy", item_ids)
    
    # Save mappings for reference
    with open(models_dir / "user_mapping.pkl", 'wb') as f:
        pickle.dump(user_mapping, f)
    
    with open(models_dir / "item_mapping.pkl", 'wb') as f:
        pickle.dump(item_mapping, f)
    
    print("Embeddings exported successfully!")
    print(f"Files saved:")
    print(f"  - {models_dir / 'user_emb.npy'}")
    print(f"  - {models_dir / 'item_emb.npy'}")
    print(f"  - {models_dir / 'item_ids.npy'}")
    
    # Verify embeddings
    print("\nVerifying embeddings...")
    loaded_user_emb = np.load(models_dir / "user_emb.npy")
    loaded_item_emb = np.load(models_dir / "item_emb.npy")
    loaded_item_ids = np.load(models_dir / "item_ids.npy")
    
    print(f"Loaded user embeddings: {loaded_user_emb.shape}")
    print(f"Loaded item embeddings: {loaded_item_emb.shape}")
    print(f"Loaded item IDs: {len(loaded_item_ids)}")
    
    # Check if embeddings are normalized
    user_norms = np.linalg.norm(loaded_user_emb, axis=1)
    item_norms = np.linalg.norm(loaded_item_emb, axis=1)
    
    print(f"User embedding norms - min: {user_norms.min():.4f}, max: {user_norms.max():.4f}")
    print(f"Item embedding norms - min: {item_norms.min():.4f}, max: {item_norms.max():.4f}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(loaded_user_emb)) or np.any(np.isinf(loaded_user_emb)):
        print("Warning: User embeddings contain NaN or infinite values!")
    
    if np.any(np.isnan(loaded_item_emb)) or np.any(np.isinf(loaded_item_emb)):
        print("Warning: Item embeddings contain NaN or infinite values!")
    
    print("Export completed successfully!")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export embeddings for serving')
    
    args = parser.parse_args()
    
    print("TunedIn Embedding Export")
    print("=" * 40)
    
    export_embeddings()


if __name__ == "__main__":
    main()

