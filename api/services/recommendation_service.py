"""
RecommendationService handles inference and training for GNN-based music recommendation models.
It supports personalized user recommendations, song similarity queries, and cold-start recommendations.
"""

import torch
from fastapi import HTTPException
import config
from loguru import logger

# Core service layer for GNN-based music recommendations
class RecommendationService:
    def __init__(self):
        self.models = {}
        self.graph = None
        self.song_id_to_idx = {}
        self.idx_to_song_id = {}
        self.song_metadata = {}
        self._initialize()

    def _initialize(self):
        # Load dataset and build graph from user-song interactions
        from utils.data_loader import load_dataset
        from utils.graph_builder import build_graph
        import os
        from models import gat, gcn, graphsage, lightgcn

        dataset = load_dataset("spotify")
        self.graph = build_graph(dataset)

        # Build index mappings and metadata for reverse lookup
        songs_df = dataset["songs"]
        self.song_id_to_idx = {song_id: i for i, song_id in enumerate(songs_df["song_id"])}
        self.idx_to_song_id = {i: song_id for song_id, i in self.song_id_to_idx.items()}

        for _, row in songs_df.iterrows():
            song_id = row["song_id"]
            self.song_metadata[song_id] = {
                "song_id": song_id,
                "song_name": row["song_name"],
                "artist_name": row["artist_name"],
                "album_name": row["album_name"],
                "genre": row["genre"],
                "popularity": row.get("popularity")
            }

        # Attempt to load pretrained models, fallback to fresh initialization
        model_classes = {
            "gcn": gcn.GCNModel,
            "gat": gat.GATModel,
            "lightgcn": lightgcn.LightGCNModel,
            "graphsage": graphsage.GraphSAGEModel
        }

        for name, cls in model_classes.items():
            path = os.path.join(config.MODELS_DIR, f"{name}_spotify.pt")
            if os.path.exists(path):
                self.models[name] = cls.load(path)
            else:
                self.models[name] = cls(
                    input_dim=None,
                    embedding_dim=config.EMBEDDING_DIM,
                    num_layers=config.NUM_GNN_LAYERS
                )

    async def recommend_for_user(self, request, model_name: str):
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        # Recommend songs for a given user by computing top-k from trained embeddings
        try:
            user_id = request.user_id
            if not user_id.startswith("user_"):
                user_id = f"user_{user_id}"

            edge_index = self.graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index
            user_idx = int(user_id.split("_")[1])

            if user_idx >= self.graph[config.USER_NODE_TYPE].num_nodes:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

            # Create a mapping of user indices to the songs they've interacted with
            user_interactions = {}
            for i in range(edge_index.size(1)):
                u_idx = edge_index[0, i].item()
                s_idx = edge_index[1, i].item()
                user_interactions.setdefault(u_idx, []).append(s_idx)

            recommendations = model.recommend(
                user_indices=torch.tensor([user_idx]),
                top_k=request.num_recommendations,
                exclude_interacted=request.exclude_listened,
                interacted_items=user_interactions if request.exclude_listened else None
            )[0]

            # Convert song indices to full metadata
            results = []
            for idx in recommendations:
                song_id = self.idx_to_song_id.get(idx)
                if song_id and song_id in self.song_metadata:
                    results.append(self.song_metadata[song_id])

            return {
                "recommendations": results,
                "model_used": model_name
            }

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    async def recommend_similar_songs(self, request, model_name: str):
        # Recommend similar songs using cosine similarity in embedding space
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        try:
            song_id = request.song_id

            if song_id not in self.song_id_to_idx:
                raise HTTPException(status_code=404, detail=f"Song {song_id} not found")

            song_idx = self.song_id_to_idx[song_id]

            # Compute dot product similarity between target song and all songs
            song_emb = model.item_embedding[song_idx].unsqueeze(0)

            # Calculate similarity with all songs
            all_song_emb = model.item_embedding
            similarities = torch.matmul(song_emb, all_song_emb.t()).squeeze()

            # Exclude the query song from results
            similarities[song_idx] = -float('inf')
            top_indices = torch.topk(similarities, k=request.num_recommendations).indices.tolist()

            # Enrich with metadata
            recommendation_items = []
            for idx in top_indices:
                similar_id = self.idx_to_song_id.get(idx)
                if similar_id and similar_id in self.song_metadata:
                    recommendation_items.append(self.song_metadata[similar_id])

            return {
                "recommendations": recommendation_items,
                "model_used": model_name
            }

        except Exception as e:
            logger.error(f"Error in recommend_similar_songs: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def recommend_from_features(self, request, model_name: str):
        # Recommend songs for new users using only input audio features (cold-start)
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        if model_name != "graphsage":
            raise HTTPException(status_code=400, detail="Only GraphSAGE model supports inductive learning for cold-start users")

        try:
            import config

            # Construct a padded feature vector for inductive embedding inference
            feature_vector = torch.zeros(config.EMBEDDING_DIM)
            for i, (feature, value) in enumerate(request.features.items()):
                if i < config.EMBEDDING_DIM:
                    feature_vector[i] = value

            user_features = feature_vector.unsqueeze(0)
            item_features = model.item_embedding

            recommendations = model.inductive_recommend(
                user_features=user_features,
                item_features=item_features,
                top_k=request.num_recommendations
            )[0]

            recommendation_items = []
            for idx in recommendations:
                song_id = self.idx_to_song_id.get(idx)
                if song_id and song_id in self.song_metadata:
                    recommendation_items.append(self.song_metadata[song_id])

            return {
                "recommendations": recommendation_items,
                "model_used": model_name
            }

        except Exception as e:
            logger.error(f"Error in recommend_from_features: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    
    async def train_model(self, model_name: str, epochs: int, learning_rate: float):
        # Train the specified model and save weights to disk
        import os   
        import config

        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        if self.graph is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")

        try:
            logger.info(f"Training model: {model_name} for {epochs} epochs")

            model = self.models[model_name]
            history = model.train(
                graph=self.graph,
                epochs=epochs,
                learning_rate=learning_rate
            )

            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_spotify.pt")
            model.save(model_path)

            return {
                "status": "success",
                "model": model_name,
                "epochs": epochs,
                "final_loss": history["loss"][-1] if history["loss"] else None
            }

        except Exception as e:
            logger.error(f"Error training model {model_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))