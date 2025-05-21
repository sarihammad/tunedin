"""
Utility function to load a GNN model class based on its name.
Used in training, evaluation, and API serving.
"""

def load_model(model_name: str):
    # Import model modules (lazy import to avoid circular dependencies)
    from models import gat, gcn, graphsage, lightgcn

    # Map string names to corresponding model classes
    model_map = {
        "gat": gat.GATModel,
        "gcn": gcn.GCNModel,
        "graphsage": graphsage.GraphSAGEModel,
        "lightgcn": lightgcn.LightGCNModel
    }

    # Validate model name
    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    # Return an instance of the selected model class
    return model_map[model_name]()