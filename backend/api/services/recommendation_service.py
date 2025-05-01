"""
RecommendationService handles inference and training for GNN-based music recommendation models.
It supports personalized user recommendations, song similarity queries, and cold-start recommendations.
"""

import torch
from fastapi import HTTPException
import backend.config as config
from loguru import logger

# Core service layer for GNN-based music recommendations
class RecommendationService:
    def __init__(self):
        """
        Initialize the RecommendationService by loading models, building the graph,
        and creating mappings for song IDs and metadata.
        """
        # Dictionary to hold different GNN models keyed by model name
        self.models = {}
        # Graph representing user-song interactions
        self.graph = None
        # Mapping from song ID to internal index
        self.song_id_to_idx = {}
        # Mapping from internal index back to song ID
        self.idx_to_song_id = {}
        # Metadata dictionary for songs keyed by song ID
        self.song_metadata = {}

        # Initialize the service by loading data and models
        self._initialize()

    def _initialize(self):
        """
        Load dataset, build the interaction graph, create index mappings,
        and load or initialize GNN models for recommendation.
        """
        # Import necessary utilities and models
        from utils.data_loader import load_dataset
        from utils.graph_builder import build_graph
        import os
        from models import gat, gcn, graphsage, lightgcn

        # Load the Spotify dataset
        dataset = load_dataset("spotify")

        # Build a heterogeneous graph from user-song interactions
        self.graph = build_graph(dataset)

        # Extract songs dataframe from dataset
        songs_df = dataset["songs"]

        # Create a mapping from song ID to a unique index for embedding lookup
        self.song_id_to_idx = {song_id: i for i, song_id in enumerate(songs_df["song_id"])}
        # Reverse mapping from index to song ID
        self.idx_to_song_id = {i: song_id for song_id, i in self.song_id_to_idx.items()}

        # Populate song metadata dictionary for detailed responses
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

        # Define model classes available for recommendation
        model_classes = {
            "gcn": gcn.GCNModel,
            "gat": gat.GATModel,
            "lightgcn": lightgcn.LightGCNModel,
            "graphsage": graphsage.GraphSAGEModel
        }

        # Attempt to load pretrained model weights, fallback to fresh initialization if not found
        for name, cls in model_classes.items():
            path = os.path.join(config.MODELS_DIR, f"{name}_spotify.pt")
            if os.path.exists(path):
                self.models[name] = cls.load(path)
            else:
                # Initialize model with default embedding dimension and number of layers
                self.models[name] = cls(
                    input_dim=None,
                    embedding_dim=config.EMBEDDING_DIM,
                    num_layers=config.NUM_GNN_LAYERS
                )

    async def recommend_for_user(self, request, model_name: str):
        """
        Generate personalized song recommendations for a user using a specified model.

        Args:
            request: Request object containing user_id, num_recommendations, and exclude_listened flag.
            model_name (str): The name of the GNN model to use for recommendations.

        Returns:
            dict: A dictionary containing recommended songs with metadata and the model used.

        Raises:
            HTTPException: If model not found, not trained, user not found, or internal errors occur.
        """
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        try:
            # Extract and normalize user ID format
            user_id = request.user_id
            if not user_id.startswith("user_"):
                user_id = f"user_{user_id}"

            # Retrieve edge index from the heterogeneous graph for user-song interactions
            edge_index = self.graph[config.USER_NODE_TYPE, config.LISTENED_EDGE, config.SONG_NODE_TYPE].edge_index
            user_idx = int(user_id.split("_")[1])

            # Validate that user index exists in the graph
            if user_idx >= self.graph[config.USER_NODE_TYPE].num_nodes:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

            # Build a dictionary mapping users to the songs they have interacted with
            user_interactions = {}
            for i in range(edge_index.size(1)):
                u_idx = edge_index[0, i].item()
                s_idx = edge_index[1, i].item()
                user_interactions.setdefault(u_idx, []).append(s_idx)

            # Generate recommendations excluding previously interacted songs if requested
            recommendations = model.recommend(
                user_indices=torch.tensor([user_idx]),
                top_k=request.num_recommendations,
                exclude_interacted=request.exclude_listened,
                interacted_items=user_interactions if request.exclude_listened else None
            )[0]

            # Convert recommended song indices to full metadata for response
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
        """
        Recommend songs similar to a given song based on embedding similarity.

        Args:
            request: Request object containing song_id and num_recommendations.
            model_name (str): The name of the GNN model to use for similarity computation.

        Returns:
            dict: A dictionary containing similar songs with metadata and the model used.

        Raises:
            HTTPException: If model or song not found, model not trained, or internal errors occur.
        """
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        try:
            song_id = request.song_id

            # Validate that the song exists in the dataset
            if song_id not in self.song_id_to_idx:
                raise HTTPException(status_code=404, detail=f"Song {song_id} not found")

            song_idx = self.song_id_to_idx[song_id]

            # Retrieve embedding vector for the target song
            song_emb = model.item_embedding[song_idx].unsqueeze(0)

            # Compute similarity scores between target song and all songs
            all_song_emb = model.item_embedding
            similarities = torch.matmul(song_emb, all_song_emb.t()).squeeze()

            # Exclude the query song from the list of similar songs
            similarities[song_idx] = -float('inf')

            # Select top-k most similar songs
            top_indices = torch.topk(similarities, k=request.num_recommendations).indices.tolist()

            # Collect metadata for each recommended similar song
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
        """
        Generate recommendations for cold-start users based solely on input audio features.

        Args:
            request: Request object containing feature dictionary and num_recommendations.
            model_name (str): The name of the GNN model to use (only 'graphsage' supports this).

        Returns:
            dict: A dictionary containing recommended songs with metadata and the model used.

        Raises:
            HTTPException: If model not found, not trained, unsupported model, or internal errors occur.
        """
        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        model = self.models[model_name]

        if not model.is_trained:
            raise HTTPException(status_code=400, detail=f"Model {model_name} is not trained")

        # Only GraphSAGE supports inductive learning for cold-start scenarios
        if model_name != "graphsage":
            raise HTTPException(status_code=400, detail="Only GraphSAGE model supports inductive learning for cold-start users")

        try:
            import backend.config as config

            # Construct a padded feature vector for the new user's audio features
            feature_vector = torch.zeros(config.EMBEDDING_DIM)
            for i, (feature, value) in enumerate(request.features.items()):
                if i < config.EMBEDDING_DIM:
                    feature_vector[i] = value

            # Prepare user and item features for inductive recommendation
            user_features = feature_vector.unsqueeze(0)
            item_features = model.item_embedding

            # Generate recommendations based on feature similarity
            recommendations = model.inductive_recommend(
                user_features=user_features,
                item_features=item_features,
                top_k=request.num_recommendations
            )[0]

            # Map recommended indices to song metadata
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
        """
        Train a specified GNN model on the user-song interaction graph.

        Args:
            model_name (str): The name of the model to train.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimizer.

        Returns:
            dict: Training status, model name, epochs, and final loss value.

        Raises:
            HTTPException: If model not found, graph not initialized, or training errors occur.
        """
        import os   
        import backend.config as config

        if model_name not in self.models:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")

        if self.graph is None:
            raise HTTPException(status_code=500, detail="Graph not initialized")

        try:
            logger.info(f"Training model: {model_name} for {epochs} epochs")

            model = self.models[model_name]
            # Train the model on the graph with specified hyperparameters
            history = model.train(
                graph=self.graph,
                epochs=epochs,
                learning_rate=learning_rate
            )

            # Save the trained model weights to disk
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