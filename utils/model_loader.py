def load_model(model_name: str):
    from models import gat, gcn, graphsage, lightgcn

    model_map = {
        "gat": gat.GATModel,
        "gcn": gcn.GCNModel,
        "graphsage": graphsage.GraphSAGEModel,
        "lightgcn": lightgcn.LightGCNModel
    }

    if model_name not in model_map:
        raise ValueError(f"Unsupported model: {model_name}")

    return model_map[model_name]()