import numpy as np

def swish(x):
    x = np.array(x)
    sigmoid = 1 / (1 + np.exp(-x))
    result = x * sigmoid
    return result.tolist()   # Convert to list if required

# Test Case
x = [0, 1, -1, 3]
print(swish(x))