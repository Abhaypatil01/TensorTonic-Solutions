def precision_recall_at_k(recommended, relevant, k):
    """
    Compute Precision@k and Recall@k.
    Returns a list: [precision, recall]
    """
    
    if k <= 0:
        raise ValueError("k must be positive")
    
    top_k = recommended[:k]
    relevant_set = set(relevant)
    
    hits = sum(1 for item in top_k if item in relevant_set)
    
    precision = hits / k
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    return [precision, recall]