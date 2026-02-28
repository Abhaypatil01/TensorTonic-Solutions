import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform to 3D point(s).
    
    Args:
        T: np.ndarray of shape (4,4)
        points: np.ndarray of shape (3,) or (N,3)
    
    Returns:
        Transformed points of shape (3,) or (N,3)
    """
    
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)
    
    single_point = False
    if points.ndim == 1:
        points = points[np.newaxis, :]
        single_point = True
    
    # Convert to homogeneous (N,4)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])
    
    # Apply transform
    transformed_h = (T @ points_h.T).T
    
    # Extract spatial coordinates
    transformed = transformed_h[:, :3]
    
    return transformed[0] if single_point else transformed