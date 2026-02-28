import numpy as np

def relu(x):
    x = np.array(x)  # Ensure it works for list, scalar, or array
    return np.maximum(0, x)

# Test Case
x = [-2, -1, 0, 3]
print(relu(x))