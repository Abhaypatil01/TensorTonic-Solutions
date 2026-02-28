import numpy as np

def dice_loss(p, y, eps=1e-8):
    p = np.array(p, dtype=float)
    y = np.array(y, dtype=float)

    # Flatten in case of 2D masks
    p = p.reshape(-1)
    y = y.reshape(-1)

    intersection = np.sum(p * y)
    dice = (2 * intersection + eps) / (np.sum(p) + np.sum(y) + eps)

    return float(1 - dice)