from functools import partial
import numpy as np

import torch
import kruskals_cpp


def get_root_pytorch(parents, node):
    # find path of objects leading to the root
    path = [node]
    root = parents[node]
    while root != path[-1]:
      path.append(root)
      root = parents[root]

    # compress the path and return
    for ancestor in path:
      parents[ancestor] = root
    return parents, root


def get_tree(edges, n):
    # Initialize weights and edges.
    weights = torch.IntTensor([1 for i in range(n)])
    parents = torch.IntTensor([i for i in range(n)])

    adj_matrix = torch.zeros(n-1, n)
    for edge in edges:
        i, j = edge
        parents, root_i = get_root_pytorch(parents, i)
        parents, root_j = get_root_pytorch(parents, j)

        if root_i != root_j:
            # Combine two forests if i and j are not in the same forest.
            heavier = max([(weights[root_i], root_i), (weights[root_j], root_j)])[1]
            for r in [root_i, root_j]:
                if r != heavier:
                    weights[heavier] = weights[heavier] + weights[r]
                    parents[r] = heavier

            # Update adjacency matrix.
            adj_matrix[i][j] = 1
            adj_matrix[j - 1][i] = 1
    return adj_matrix


def kruskals_pytorch(weights_and_edges, n):
    """Non-batched implementation of kruskal's algorithm in Pytorch.

    Easier to undertand than the batched version, but equivalent.
    Args:
        weights_and_edges: Shape (n * (n - 1) / 2, 3), where
            weights_and_edges[i] = [weight_i, node1_i, node2_i] for edge i.
        n: Number of nodes.
    Returns:
        Adjacency matrix with diagonal removed. Shape (n - 1, n)
    """ 

    # Sort edges based on weights, in descending order.
    sorted_edges = weights_and_edges[
        (-1.0 * weights_and_edges[:,0]).argsort(), 1:].int()
    return get_tree(sorted_edges, n)


def get_root_pytorch_batched(parents, node, n):
    bs = parents.size(0)
    arange = torch.arange(bs)
    # Find path of nodes leading to the root.
    path = torch.zeros_like(parents)
    path[:, 0] = node
    root = parents[torch.arange(bs), node]
    for i in range(1, n):
        path[:, i] = root
        root = parents[torch.arange(bs), root]
    # Compress the path and return.
    for i in range(1, n):
        parents[arange, path[:, i]] = root
    return parents, root


def kruskals_pytorch_batched(weights_and_edges, n):
    """Batched kruskal's algorithm in Pytorch.

    Args:
        weights_and_edges: Shape (batch size, n * (n - 1) / 2, 3), where
            weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i.
        n: Number of nodes.
    Returns:
        Adjacency matrix with diagonal removed. Shape (batch size, n-1, n)
    """
    device = weights_and_edges.device
    batch_size = weights_and_edges.size(0)
    arange = torch.arange(batch_size)
    # Sort edges based on weights, in descending order.
    sorted_weights = torch.argsort(
        weights_and_edges[:, :, 0], -1, descending=True)
    dummy = sorted_weights.unsqueeze(2).expand(
        *(sorted_weights.shape + (weights_and_edges.size(2),)))
    sorted_edges = torch.gather(weights_and_edges, 1, dummy)[:, :, 1:]
    sorted_edges = sorted_edges.transpose(0, 1).long()

    # Initialize weights and edges.
    weights = torch.ones((batch_size, n)).to(device)
    parents = torch.arange(n).repeat((batch_size, 1)).to(device)

    adj_matrix = torch.zeros((batch_size, n - 1, n)).to(device)
    for edge in sorted_edges:
        i, j = edge.transpose(0, 1)
        parents, root_i = get_root_pytorch_batched(parents, i, n)
        parents, root_j = get_root_pytorch_batched(parents, j, n)
        is_i_and_j_not_in_same_forest = (root_i != root_j).int()

        # Combine two forests if i and j are not in the same forest.
        is_i_heavier_than_j = (
            weights[arange, root_i] > weights[arange, root_j]).int()
        weights_root_i = weights[arange, root_i] + (
            (weights[arange, root_j] * is_i_heavier_than_j)
            * is_i_and_j_not_in_same_forest +
            0.0 * (1.0 - is_i_and_j_not_in_same_forest))
        parents_root_i = (
            (root_i * is_i_heavier_than_j +  root_j * (1 - is_i_heavier_than_j)) 
            * is_i_and_j_not_in_same_forest +
            root_i * (1 - is_i_and_j_not_in_same_forest))
        weights_root_j = weights[arange, root_j] + (
            weights[arange, root_i] * (1 - is_i_heavier_than_j) 
            * is_i_and_j_not_in_same_forest +
            0.0 * (1.0 - is_i_and_j_not_in_same_forest))
        parents_root_j = (
            (root_i * is_i_heavier_than_j +  root_j * (1 - is_i_heavier_than_j)) 
            * is_i_and_j_not_in_same_forest +
            root_j * (1 - is_i_and_j_not_in_same_forest))
        weights[arange, root_i] = weights_root_i
        weights[arange, root_j] = weights_root_j
        parents[arange, root_i] = parents_root_i
        parents[arange, root_j] = parents_root_j

        # Update adjacency matrix.
        adj_matrix[arange, i, j] = is_i_and_j_not_in_same_forest.float()
        adj_matrix[arange, j - 1, i] = is_i_and_j_not_in_same_forest.float()
    return adj_matrix


def kruskals_cpp_pytorch(weights_and_edges, n):
    """Kruskal's algorithm in C++.

    Does the sorting in Pytorch, then the equivalent of the get_tree in Pytorch
    above, in C++.

    Args:
        weights_and_edges: Shape (batch size, n * (n - 1) / 2, 3), where
            weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i.
        n: Number of nodes.
    Returns:
        Adjacency matrix with diagonal removed. Shape (batch size, n-1, n)
    """
    # Sort edges based on weights, in descending order.
    sorted_weights = torch.argsort(
        weights_and_edges[:, :, 0], -1, descending=True)
    dummy = sorted_weights.unsqueeze(2).expand(
        *(sorted_weights.shape + (weights_and_edges.size(2),)))
    # sorted_edges is shape (batch_size, n * (n - 1) / 2, 2)
    sorted_edges = torch.gather(weights_and_edges, 1, dummy)[:, :, 1:]

    adj_matrix = kruskals_cpp.get_tree(sorted_edges.int(), n)
    return adj_matrix


def kruskals_cpp_pytorch2(weights_and_edges, n):
    """Kruskal's algorithm in C++.

    A different implementation of Kruskal's algorithm than all the above
    implementations.

    Args:
        weights_and_edges: Shape (batch size, n * (n - 1) / 2, 3), where
            weights_and_edges[.][i] = [weight_i, node1_i, node2_i] for edge i.
        n: Number of nodes.
    Returns:
        Adjacency matrix with diagonal removed. Shape (batch size, n-1, n)
    """
    adj_matrix = kruskals_cpp.kruskals(weights_and_edges, n)
    return adj_matrix


if __name__ == "__main__":
    # Run below tests with "python -m core.kruskals" from main folder.
    # Check equivalence.
    n = 4
    weights = np.array([0.7601073, -0.20460297, -0.4689217, 
                        -0.5127163, -1.9022679, 1.1506207])
    vertices = np.triu_indices(n=n-1, m=n, k=1)
    weights_and_edges = np.array(
        [list(e) for e in zip(weights, vertices[0], vertices[1])])

    res_pytorch = kruskals_pytorch(torch.Tensor(weights_and_edges), n)
    res_pytorch_batched = kruskals_pytorch_batched(
        torch.Tensor(weights_and_edges).unsqueeze(0), n)
    res_cpp = kruskals_cpp_pytorch(
        torch.Tensor(weights_and_edges).unsqueeze(0), n)

    soln = np.array([[1., 1., 1., 0.],
                     [1., 0., 0., 0.],
                     [0., 0., 1., 1.]])

    np.testing.assert_almost_equal(res_pytorch, soln)
    np.testing.assert_almost_equal(res_pytorch_batched.squeeze(0), soln)
    np.testing.assert_almost_equal(res_cpp.squeeze(0), soln)

    # Also check equivalence on batch of random weights.
    n = 7
    bs = 100
    weights = np.random.rand(bs, int(n * (n - 1) / 2))

    # Pytorch.
    weights = torch.tensor(weights).float()
    vertices = torch.triu_indices(n-1, n, offset=1)
    tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
    weights_and_edges = torch.cat([weights.unsqueeze(-1), tiled_vertices], axis=-1)
    # Non-batched Pytorch.
    res_pytorch = []
    for i in range(bs):
        res_pytorch.append(kruskals_pytorch(weights_and_edges[i], n))
    res_pytorch = torch.stack(res_pytorch).numpy()
    # Batched Pytorch.
    res_pytorch_batched = kruskals_pytorch_batched(weights_and_edges, n).numpy()

    # C++ (with pytorch).
    res_cpp_pytorch = kruskals_cpp_pytorch(weights_and_edges, n).numpy()

    # C++ (with pytorch2).
    res_cpp_pytorch2 = kruskals_cpp_pytorch(weights_and_edges, n).numpy()

    np.testing.assert_almost_equal(res_pytorch, res_pytorch_batched)
    np.testing.assert_almost_equal(res_pytorch_batched, res_cpp_pytorch)
    np.testing.assert_almost_equal(res_cpp_pytorch, res_cpp_pytorch2)
