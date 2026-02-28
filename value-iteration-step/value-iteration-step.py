import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one value iteration step.
    Returns a Python list.
    """
    
    V = np.asarray(values, dtype=float)
    T = np.asarray(transitions, dtype=float)
    R = np.asarray(rewards, dtype=float)
    
    # Correct contraction over next-state dimension
    expected_future = np.einsum('sap,p->sa', T, V)
    
    Q = R + gamma * expected_future
    
    V_new = np.max(Q, axis=1)
    
    return V_new.tolist()