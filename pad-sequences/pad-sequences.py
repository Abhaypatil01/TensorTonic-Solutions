import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L)
    """
    
    # Convert to list (handles np.array input)
    seqs = list(seqs)
    N = len(seqs)
    
    # Determine max length
    if max_len is None:
        L = 0
        for seq in seqs:
            if len(seq) > L:
                L = len(seq)
    else:
        L = int(max_len)
    
    # Create padded array
    padded = np.full((N, L), pad_value)
    
    # Fill values
    for i in range(N):
        seq = seqs[i]
        length = min(len(seq), L)
        if length > 0:
            padded[i, :length] = seq[:length]
    
    return padded