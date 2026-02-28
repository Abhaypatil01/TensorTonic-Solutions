import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    """
    Compute sinusoidal positional encoding matrix.
    
    Returns:
        np.ndarray of shape (seq_len, d_model)
    """
    
    # Positions (seq_len, 1)
    positions = np.arange(seq_len, dtype=float)[:, np.newaxis]
    
    # Compute dimension indices for all columns
    dims = np.arange(d_model, dtype=float)
    
    # Compute angle rates
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    
    # Compute angles (broadcasting)
    angles = positions * angle_rates
    
    # Initialize output
    pe = np.zeros((seq_len, d_model), dtype=float)
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    
    # Apply cos to odd indices
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    
    return pe