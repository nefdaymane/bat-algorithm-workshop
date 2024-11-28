def sphere_function(position):
    """Sphere function: f(x, y) = x^2 + y^2."""
    x, y = position
    return x**2 + y**2


def qap_objective(permutation, distance_matrix, flow_matrix):
    """
    Compute the QAP objective value for a given permutation.
    
    Args:
        permutation (list): A permutation of facility assignments.
        distance_matrix (np.ndarray): Distance matrix.
        flow_matrix (np.ndarray): Flow matrix.
    
    Returns:
        int: Objective function value representing the total cost.
    """
    size = len(permutation)
    cost = 0
    for i in range(size):
        for j in range(size):
            cost += flow_matrix[i][j] * distance_matrix[permutation[i]][permutation[j]]
    return cost

