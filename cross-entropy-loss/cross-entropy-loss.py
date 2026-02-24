import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred, dtype=float)
    
    N = y_true.shape[0]
    
    # Select probability of the correct class for each sample
    correct_probs = y_pred[np.arange(N), y_true]
    
    # Compute average cross-entropy
    loss = -np.mean(np.log(correct_probs))
    
    return loss