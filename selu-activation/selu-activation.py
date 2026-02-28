import numpy as np

def selu(x):
    x = np.array(x)
    
    lam = 1.0507
    alpha = 1.6733
    
    return np.where(
        x > 0,
        lam * x,
        lam * alpha * (np.exp(x) - 1)
    )

# Test Case
x = [1, -1, 0]
print(selu(x))
