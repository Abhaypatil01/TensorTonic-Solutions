import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)

    # Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature

    # Numerical stability: subtract row max
    S_max = np.max(S, axis=1, keepdims=True)
    S = S - S_max

    exp_S = np.exp(S)

    # Positive pairs are diagonal elements
    positives = np.diag(exp_S)

    # Denominator: sum over rows
    denominator = np.sum(exp_S, axis=1)

    loss = -np.log(positives / denominator)

    return float(np.mean(loss))