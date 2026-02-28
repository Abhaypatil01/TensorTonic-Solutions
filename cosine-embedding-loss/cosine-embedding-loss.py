import numpy as np

def cosine_embedding_loss(x1, x2, label, margin=0.0):
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)

    # Cosine similarity
    dot = np.dot(x1, x2)
    norm1 = np.linalg.norm(x1)
    norm2 = np.linalg.norm(x2)
    
    cosine = dot / (norm1 * norm2)

    # Loss based on label
    if label == 1:
        loss = 1 - cosine
    elif label == -1:
        loss = max(0.0, cosine - margin)
    else:
        raise ValueError("label must be +1 or -1")

    return float(loss)