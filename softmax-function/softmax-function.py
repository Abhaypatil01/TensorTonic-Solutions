import numpy as np

def softmax(x):
    x = np.array(x)
    
    # If input is 1D
    if x.ndim == 1:
        x_shifted = x - np.max(x)
        exp_values = np.exp(x_shifted)
        probabilities = exp_values / np.sum(exp_values)
    
    # If input is 2D
    else:
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_values = np.exp(x_shifted)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    return probabilities.tolist()