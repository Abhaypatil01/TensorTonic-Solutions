import numpy as np
from numpy import tanh

def tanh_activation(x):
    x = np.array(x)
    return tanh(x)

x = [0, 1, -1, 3]
print(tanh_activation(x))