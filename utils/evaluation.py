"""
Evaluation metrics for recommendation systems.
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
from loguru import logger

def precision_at_k(y_true, y_pred, k=10):
    """
    Calculate precision@k for recommendations.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@k
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    precisions = []
    for true_items, pred_items in zip(y_true, y_pred):
        # Take only the top-k predictions
        pred_items_k = pred_items[:k]
        
        # Calculate precision
        if len(pred_items_k) == 0:
            precisions.append(0.0)
        else:
            # Count the number of true items in the predictions
            n_relevant = sum(1 for item in pred_items_k if item in true_items)
            precisions.append(n_relevant / len(pred_items_k))
    
    return np.mean(precisions)

def recall_at_k(y_true, y_pred, k=10):
    """
    Calculate recall@k for recommendations.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@k
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    recalls = []
    for true_items, pred_items in zip(y_true, y_pred):
        # Take only the top-k predictions
        pred_items_k = pred_items[:k]
        
        # Calculate recall
        if len(true_items) == 0:
            recalls.append(1.0)  # If there are no true items, recall is 1
        else:
            # Count the number of true items in the predictions
            n_relevant = sum(1 for item in pred_items_k if item in true_items)
            recalls.append(n_relevant / len(true_items))
    
    return np.mean(recalls)

def ndcg_at_k(y_true, y_pred, k=10):
    """
    Calculate NDCG@k for recommendations.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: NDCG@k
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Convert to binary relevance format for sklearn
    y_true_binary = []
    y_pred_scores = []
    
    for true_items, pred_items in zip(y_true, y_pred):
        # Take only the top-k predictions
        pred_items_k = pred_items[:k]
        
        # Create binary relevance vector
        binary_relevance = np.zeros(len(pred_items_k))
        for i, item in enumerate(pred_items_k):
            if item in true_items:
                binary_relevance[i] = 1
        
        # Create prediction scores (higher rank = higher score)
        pred_scores = np.array([len(pred_items_k) - i for i in range(len(pred_items_k))])
        
        y_true_binary.append(binary_relevance)
        y_pred_scores.append(pred_scores)
    
    # Calculate NDCG
    return ndcg_score(y_true_binary, y_pred_scores)

def map_at_k(y_true, y_pred, k=10):
    """
    Calculate Mean Average Precision (MAP)@k for recommendations.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: MAP@k
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    aps = []
    for true_items, pred_items in zip(y_true, y_pred):
        # Take only the top-k predictions
        pred_items_k = pred_items[:k]
        
        # Calculate average precision
        if len(true_items) == 0:
            aps.append(0.0)
            continue
        
        hits = 0
        sum_precisions = 0
        
        for i, item in enumerate(pred_items_k):
            if item in true_items:
                hits += 1
                sum_precisions += hits / (i + 1)
        
        if hits == 0:
            aps.append(0.0)
        else:
            aps.append(sum_precisions / min(len(true_items), k))
    
    return np.mean(aps)

def hit_rate_at_k(y_true, y_pred, k=10):
    """
    Calculate Hit Rate@k for recommendations.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        
    Returns:
        float: Hit Rate@k
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    hits = []
    for true_items, pred_items in zip(y_true, y_pred):
        # Take only the top-k predictions
        pred_items_k = pred_items[:k]
        
        # Check if at least one true item is in the predictions
        hit = any(item in true_items for item in pred_items_k)
        hits.append(1.0 if hit else 0.0)
    
    return np.mean(hits)

def evaluate_recommendations(y_true, y_pred, k=10, metrics=None):
    """
    Evaluate recommendations using multiple metrics.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        k (int): Number of recommendations to consider
        metrics (list): List of metrics to calculate
        
    Returns:
        dict: Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ["precision", "recall", "ndcg", "map", "hit_rate"]
    
    results = {}
    
    for metric in metrics:
        if metric == "precision":
            results["precision@k"] = precision_at_k(y_true, y_pred, k)
        elif metric == "recall":
            results["recall@k"] = recall_at_k(y_true, y_pred, k)
        elif metric == "ndcg":
            results["ndcg@k"] = ndcg_at_k(y_true, y_pred, k)
        elif metric == "map":
            results["map@k"] = map_at_k(y_true, y_pred, k)
        elif metric == "hit_rate":
            results["hit_rate@k"] = hit_rate_at_k(y_true, y_pred, k)
    
    return results

def evaluate_cold_start(y_true, y_pred, user_counts, item_counts, k=10, threshold=5):
    """
    Evaluate recommendations for cold-start users and items.
    
    Args:
        y_true (list): List of lists of true items
        y_pred (list): List of lists of predicted items
        user_counts (dict): Dictionary of user IDs to interaction counts
        item_counts (dict): Dictionary of item IDs to interaction counts
        k (int): Number of recommendations to consider
        threshold (int): Threshold for cold-start (users/items with fewer interactions)
        
    Returns:
        dict: Dictionary of metric names and values for cold-start scenarios
    """
    # Identify cold-start users and items
    cold_start_users = {user for user, count in user_counts.items() if count < threshold}
    cold_start_items = {item for item, count in item_counts.items() if count < threshold}
    
    # Filter recommendations for cold-start users
    cold_start_user_indices = [i for i, user in enumerate(user_counts.keys()) if user in cold_start_users]
    
    if cold_start_user_indices:
        cold_start_y_true = [y_true[i] for i in cold_start_user_indices]
        cold_start_y_pred = [y_pred[i] for i in cold_start_user_indices]
        
        # Evaluate cold-start user recommendations
        cold_start_user_results = evaluate_recommendations(cold_start_y_true, cold_start_y_pred, k)
        cold_start_user_results = {f"cold_start_user_{k}": v for k, v in cold_start_user_results.items()}
    else:
        cold_start_user_results = {f"cold_start_user_{k}": 0.0 for k in ["precision@k", "recall@k", "ndcg@k", "map@k", "hit_rate@k"]}
    
    # Filter recommendations containing cold-start items
    cold_start_item_results = {}
    
    # Count how many cold-start items are recommended
    cold_start_item_hit_rate = []
    for pred_items in y_pred:
        pred_items_k = pred_items[:k]
        cold_start_items_recommended = sum(1 for item in pred_items_k if item in cold_start_items)
        cold_start_item_hit_rate.append(cold_start_items_recommended > 0)
    
    cold_start_item_results["cold_start_item_hit_rate@k"] = np.mean(cold_start_item_hit_rate)
    
    # Combine results
    results = {}
    results.update(cold_start_user_results)
    results.update(cold_start_item_results)
    
    return results 