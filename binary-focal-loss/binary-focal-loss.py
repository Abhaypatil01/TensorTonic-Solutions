import numpy as np

def binary_focal_loss(predictions, targets, alpha=0.25, gamma=2.0, eps=1e-12):
    predictions = np.array(predictions, dtype=float)
    targets = np.array(targets, dtype=float)

    # Numerical stability
    predictions = np.clip(predictions, eps, 1 - eps)

    # Compute pt
    pt = np.where(targets == 1, predictions, 1 - predictions)

    # Compute focal loss
    loss = -alpha * (1 - pt) ** gamma * np.log(pt)

    return float(np.mean(loss))