import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    G = np.array(G, dtype=float)

    # Step 1: accumulate squared gradients
    new_G = G + g**2

    # Step 2: update parameters (eps inside sqrt)
    new_w = w - lr * g / np.sqrt(new_G + eps)

    return new_w, new_G