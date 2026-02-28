import numpy as np

def entropy_node(y):
    """
    Compute entropy of class labels in a node.
    """
    
    y = np.asarray(y)
    
    if y.size == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    
    # Entropy formula
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)