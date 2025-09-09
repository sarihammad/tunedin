"""Build FAISS index for efficient similarity search."""

import argparse
import faiss
import numpy as np
from pathlib import Path


def build_faiss_index(index_type="hnsw", nlist=100, m=16):
    """Build FAISS index from item embeddings."""
    print("Loading item embeddings...")
    
    models_dir = Path("models")
    
    # Load item embeddings
    item_emb = np.load(models_dir / "item_emb.npy")
    item_ids = np.load(models_dir / "item_ids.npy")
    
    print(f"Item embeddings shape: {item_emb.shape}")
    print(f"Item IDs shape: {item_ids.shape}")
    
    # Ensure embeddings are float32
    item_emb = item_emb.astype(np.float32)
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(item_emb)
    
    # Create FAISS index
    dimension = item_emb.shape[1]
    print(f"Building FAISS index with dimension {dimension}...")
    
    if index_type == "hnsw":
        # HNSW index for better recall
        index = faiss.IndexHNSWFlat(dimension, m)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        print(f"Created HNSW index with m={m}")
        
    elif index_type == "ivf-pq":
        # IVF-PQ index for memory efficiency
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)
        print(f"Created IVF-PQ index with nlist={nlist}")
        
    elif index_type == "flat":
        # Flat index for exact search
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        print("Created Flat index")
        
    else:
        raise ValueError(f"Unknown index type: {index_type}")
    
    # Train index if needed
    if hasattr(index, 'is_trained') and not index.is_trained:
        print("Training index...")
        index.train(item_emb)
    
    # Add embeddings to index
    print("Adding embeddings to index...")
    index.add(item_emb)
    
    print(f"Index built with {index.ntotal} vectors")
    
    # Save index
    index_path = models_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Index saved to {index_path}")
    
    # Test the index
    print("\nTesting index...")
    test_query = item_emb[0:1]  # Use first item as query
    
    # Search for top-10 similar items
    k = 10
    scores, indices = index.search(test_query, k)
    
    print(f"Query item ID: {item_ids[0]}")
    print("Top-10 similar items:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # Valid result
            print(f"  {i+1}. Item {item_ids[idx]} (score: {score:.4f})")
    
    # Get index statistics
    print(f"\nIndex statistics:")
    print(f"  - Total vectors: {index.ntotal}")
    print(f"  - Dimension: {index.d}")
    print(f"  - Index type: {type(index).__name__}")
    
    if hasattr(index, 'hnsw'):
        print(f"  - HNSW M: {index.hnsw.M}")
        print(f"  - HNSW efConstruction: {index.hnsw.efConstruction}")
        print(f"  - HNSW efSearch: {index.hnsw.efSearch}")
    
    if hasattr(index, 'nlist'):
        print(f"  - IVF nlist: {index.nlist}")
    
    print("FAISS index built successfully!")


def main():
    """Main function to build FAISS index."""
    parser = argparse.ArgumentParser(description='Build FAISS index for item embeddings')
    parser.add_argument('--index-type', type=str, default='hnsw', 
                       choices=['hnsw', 'ivf-pq', 'flat'],
                       help='Type of FAISS index to build')
    parser.add_argument('--nlist', type=int, default=100,
                       help='Number of clusters for IVF-PQ index')
    parser.add_argument('--m', type=int, default=16,
                       help='HNSW parameter M (number of bi-directional links)')
    
    args = parser.parse_args()
    
    print("TunedIn FAISS Index Builder")
    print("=" * 40)
    
    build_faiss_index(
        index_type=args.index_type,
        nlist=args.nlist,
        m=args.m
    )


if __name__ == "__main__":
    main()

