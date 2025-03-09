"""
Tests for the TunedIn models.
"""
import os
import sys
import unittest
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.data_loader import create_mock_spotify_dataset
from utils.graph_builder import build_graph
from models.gcn import GCNModel
from models.gat import GATModel
from models.lightgcn import LightGCNModel
from models.graphsage import GraphSAGEModel

class TestModels(unittest.TestCase):
    """
    Test cases for the TunedIn models.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test data and models.
        """
        # Create a small mock dataset
        cls.dataset = create_mock_spotify_dataset()
        
        # Build graph
        cls.graph = build_graph(cls.dataset)
        
        # Initialize models
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
            # Forward pass
            embeddings = model(self.graph)
            
            # Check if embeddings are returned
            self.assertIsNotNone(embeddings)
            
            if model_name in ["gcn", "gat", "graphsage"]:
                # Check if user and song embeddings are present
                self.assertIn(config.USER_NODE_TYPE, embeddings)
                self.assertIn(config.SONG_NODE_TYPE, embeddings)
                
                # Check embedding dimensions
                self.assertEqual(embeddings[config.USER_NODE_TYPE].size(1), 32)
                self.assertEqual(embeddings[config.SONG_NODE_TYPE].size(1), 32)
            else:  # LightGCN
                # Check if user and song embeddings are returned as a tuple
                self.assertEqual(len(embeddings), 2)
                
                # Check embedding dimensions
                self.assertEqual(embeddings[0].size(1), 32)
                self.assertEqual(embeddings[1].size(1), 32)
    
    def test_training(self):
        """
        Test training of models.
        """
        for model_name, model in self.models.items():
            # Train for a few epochs
            history = model.train(self.graph, epochs=2, batch_size=128)
            
            # Check if history is returned
            self.assertIsNotNone(history)
            
            # Check if loss is recorded
            self.assertIn("loss", history)
            self.assertEqual(len(history["loss"]), 2)
            
            # Check if model is marked as trained
            self.assertTrue(model.is_trained)
    
    def test_recommendations(self):
        """
        Test recommendation generation.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Generate recommendations for a few users
            user_indices = torch.tensor([0, 1, 2])
            recommendations = model.recommend(
                user_indices=user_indices,
                top_k=5
            )
            
            # Check if recommendations are returned
            self.assertIsNotNone(recommendations)
            
            # Check if recommendations are returned for each user
            self.assertEqual(len(recommendations), len(user_indices))
            
            # Check if the correct number of recommendations is returned
            for user_recs in recommendations:
                self.assertEqual(len(user_recs), 5)
    
    def test_evaluation(self):
        """
        Test model evaluation.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Evaluate model
            metrics = model.evaluate(self.graph, k=5)
            
            # Check if metrics are returned
            self.assertIsNotNone(metrics)
            
            # Check if standard metrics are included
            self.assertIn("precision@k", metrics)
            self.assertIn("recall@k", metrics)
            self.assertIn("ndcg@k", metrics)
    
    def test_save_load(self):
        """
        Test saving and loading models.
        """
        for model_name, model in self.models.items():
            # Train model if not already trained
            if not model.is_trained:
                model.train(self.graph, epochs=2, batch_size=128)
            
            # Save model
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}_test.pt")
            model.save(model_path)
            
            # Check if file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Load model
            if model_name == "gcn":
                loaded_model = GCNModel.load(model_path)
            elif model_name == "gat":
                loaded_model = GATModel.load(model_path)
            elif model_name == "lightgcn":
                loaded_model = LightGCNModel.load(model_path)
            elif model_name == "graphsage":
                loaded_model = GraphSAGEModel.load(model_path)
            
            # Check if model is loaded correctly
            self.assertIsNotNone(loaded_model)
            self.assertEqual(loaded_model.embedding_dim, model.embedding_dim)
            self.assertTrue(loaded_model.is_trained)
            
            # Clean up
            os.remove(model_path)

if __name__ == "__main__":
    unittest.main() 