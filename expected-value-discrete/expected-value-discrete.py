import numpy as np

def expected_value_discrete(x, p):
    """
    Compute expected value of a discrete random variable.
    Raises ValueError if probabilities are invalid.
    """
    
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    
    # Check same length
    if x.shape != p.shape:
        raise ValueError("x and p must have same shape")
    
    # Check probabilities are valid
    if np.any(p < 0) or not np.isclose(np.sum(p), 1.0):
        raise ValueError("Probabilities must be non-negative and sum to 1")
    
    return float(np.sum(x * p))