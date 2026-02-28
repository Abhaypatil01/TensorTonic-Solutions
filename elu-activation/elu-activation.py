import numpy as np

def elu(x, alpha=1.0):
    x = np.array(x)
    result = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return result.tolist()   # ✅ Convert to list

# Test Case
x = [1, -1, 0, 2, -0.5]
print(elu(x))