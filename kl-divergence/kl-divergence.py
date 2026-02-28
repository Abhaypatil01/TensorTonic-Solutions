import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Add epsilon for numerical stability
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    
    return float(np.sum(p * np.log(p / q)))