"""
Tests for the TunedIn models.
"""
import os
import sys
import unittest
import torch
import numpy as np
import tempfile  # Added import for temporary file handling

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_loader import create_mock_spotify_dataset
from utils.graph_builder import build_graph
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel

# This test suite verifies the forward pass, training, evaluation, recommendation generation,
# and persistence (save/load) functionalities of all TunedIn models.
class TestModels(unittest.TestCase):
    """
    Test cases for the TunedIn models.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test data and models.
        """
        # Create a small mock dataset to simulate Spotify data
        cls.dataset = create_mock_spotify_dataset()
        
        # Build graph structure from the dataset
        cls.graph = build_graph(cls.dataset)
        
        # Initialize models with embedding dimension 32 and input_dim as None
        cls.models = {
            "gcn": GCNModel(input_dim=None, embedding_dim=32),
            "gat": GATModel(input_dim=None, embedding_dim=32),
            "lightgcn": LightGCNModel(input_dim=None, embedding_dim=32),
            "graphsage": GraphSAGEModel(input_dim=None, embedding_dim=32)
        }
    
    def test_forward_pass(self):
        """
        Test forward pass of models.
        """
        for model_name, model in self.models.items():
            # Perform forward pass to get embeddings
            embeddings = model(self.graph)
            
            # Assert embeddings are returned and not None
            self.assertIsNotNone(embeddings)
            
            if model_name in ["gcn", "gat", "graphsage"]:
                # For these models, embeddings should be a dict containing user and song embeddings
                self.assertIn(config.USER_NODE_TYPE, embeddings)
                self.assertIn(config.SONG_NODE_TYPE, embeddings)
                
                # Check that embeddings have the correct dimension (32)
                self.assertEqual(embeddings[config.USER_NODE_TYPE].size(1), 32)
                self.assertEqual(embeddings[config.SONG_NODE_TYPE].size(1), 32)
            else:  # LightGCN returns a tuple of user and song embeddings
                # Assert that embeddings is a tuple of length 2
                self.assertEqual(len(embeddings), 2)
                
                # Check that each embedding tensor has the correct dimension (32)
                self.assertEqual(embeddings[0].size(1), 32)
                self.assertEqual(embeddings[1].size(1), 32)
    
    def test_training(self):
        """
        Test training of models.
        """
        for model_name, model in self.models.items():
            # Train model for 2 epochs with batch size 128
            history = model.train(self.graph, epochs=2, batch_size=128)
            
            # Assert training history is returned and not None
            self.assertIsNotNone(history)
            
            # Assert 'loss' is recorded in history and has length equal to number of epochs
            self.assertIn("loss", history)
            self.assertEqual(len(history["loss"]), 2)
            
            # Assert model is marked as trained after training
            self.assertTrue(model.is_trained)
    
    def test_recommendations(self):
        """
        Test recommendation generation.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained to ensure recommendations can be generated
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Define user indices to generate recommendations for
            user_indices = torch.tensor([0, 1, 2])
            
            # Generate top 5 recommendations per user
            recommendations = model.recommend(
                user_indices=user_indices,
                top_k=5
            )
            
            # Assert recommendations are returned and not None
            self.assertIsNotNone(recommendations)
            
            # Assert recommendations are returned for each user requested
            self.assertEqual(len(recommendations), len(user_indices))
            
            # Assert each user's recommendations list contains exactly 5 items
            for user_recs in recommendations:
                self.assertEqual(len(user_recs), 5)
    
    def test_evaluation(self):
        """
        Test model evaluation.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained to enable evaluation
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Evaluate model with top-k = 5
            metrics = model.evaluate(self.graph, k=5)
            
            # Assert evaluation metrics are returned and not None
            self.assertIsNotNone(metrics)
            
            # Assert standard evaluation metrics are included in the results
            self.assertIn("precision@k", metrics)
            self.assertIn("recall@k", metrics)
            self.assertIn("ndcg@k", metrics)
    
    def test_save_load(self):
        """
        Test saving and loading models.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained to ensure it can be saved
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Use a temporary file to save the model to avoid side effects
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                model_path = tmp_file.name
            
            # Save model to temporary file
            model.save(model_path)
            
            # Assert the saved file exists on disk
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model from the saved file according to its type
            if model_name == "gcn":
                loaded_model = GCNModel.load(model_path)
            elif model_name == "gat":
                loaded_model = GATModel.load(model_path)
            elif model_name == "lightgcn":
                loaded_model = LightGCNModel.load(model_path)
            elif model_name == "graphsage":
                loaded_model = GraphSAGEModel.load(model_path)
            
            # Assert the loaded model is not None
            self.assertIsNotNone(loaded_model)
            
            # Assert the loaded model has the same embedding dimension as the original
            self.assertEqual(loaded_model.embedding_dim, model.embedding_dim)
            
            # Assert the loaded model is marked as trained
            self.assertTrue(loaded_model.is_trained)
            
            # Clean up by removing the temporary saved model file
            os.remove(model_path)

if __name__ == "__main__":
    unittest.main()