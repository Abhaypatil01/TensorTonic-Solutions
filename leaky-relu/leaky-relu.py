import numpy as np

def leaky_relu(x, alpha=0.01):
    x = np.array(x)
    return np.where(x >= 0, x, alpha * x)

# Test Case
x = [-2, -1, 0, 1, 2]
print(leaky_relu(x))