import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)

    # Handle single pair case
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
        y = y.reshape(-1)

    # Euclidean distance
    d = np.linalg.norm(a - b, axis=1)

    # Per-sample loss
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2

    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        return loss.tolist()