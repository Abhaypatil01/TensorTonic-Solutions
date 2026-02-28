import numpy as np

def focal_loss(p, y, gamma=2.0):
    p = np.array(p)
    y = np.array(y)

    # Compute focal loss per sample
    loss = (
        - (1 - p) ** gamma * y * np.log(p)
        - p ** gamma * (1 - y) * np.log(1 - p)
    )

    return float(np.mean(loss))