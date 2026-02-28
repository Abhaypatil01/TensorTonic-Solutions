import numpy as np
from math import erf, sqrt

def gelu(x):
    x = np.array(x)
    return 0.5 * x * (1 + np.vectorize(erf)(x / sqrt(2)))

# Test Case
x = np.array([-1, 0, 1])
print(gelu(x))