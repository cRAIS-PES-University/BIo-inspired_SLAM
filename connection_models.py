import numpy as np

def generate_sparse_connections(pre_size, post_size, sparsity=0.1, weight_mean=0.3, weight_std=0.1):
    """
    Generate sparse synaptic connections between pre- and post-synaptic neurons.

    Parameters:
    - pre_size: int, number of presynaptic neurons
    - post_size: int, number of postsynaptic neurons
    - sparsity: float, proportion of active connections relative to total possible connections
    - weight_mean: float, mean value of synaptic weights
    - weight_std: float, standard deviation of synaptic weights

    Returns:
    - connections: ndarray, synaptic weight matrix
    """
    num_connections = int(pre_size * post_size * sparsity)
    pre_indices = np.random.randint(0, pre_size, size=num_connections)
    post_indices = np.random.randint(0, post_size, size=num_connections)
    weights = np.random.normal(loc=weight_mean, scale=weight_std, size=num_connections)

    connection_matrix = np.zeros((pre_size, post_size))
    connection_matrix[pre_indices, post_indices] = weights
    return connection_matrix

def generate_grouped_connections(pre_size, post_size, num_groups, sparsity=0.1, weight_mean=0.3, weight_std=0.1):
    """
    Generate synaptic connections with neurons grouped into clusters.

    Parameters:
    - pre_size, post_size, sparsity, weight_mean, weight_std: same as generate_sparse_connections
    - num_groups: int, number of distinct neuron groups or clusters

    Returns:
    - connections: ndarray, matrix of synaptic weights with grouped connectivity patterns
    """
    connections = np.zeros((pre_size, post_size))
    group_size = pre_size // num_groups
    for g in range(num_groups):
        group_start = g * group_size
        group_end = group_start + group_size if g < num_groups - 1 else pre_size
        connections[group_start:group_end, :] = generate_sparse_connections(
            group_end - group_start, post_size, sparsity, weight_mean, weight_std)
    return connections
