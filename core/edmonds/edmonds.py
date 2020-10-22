from functools import partial
import networkx as nx
import numpy as np

import torch
import edmonds_cpp


def edmonds_python(adjs, n):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        adjs: shape (batch_size, n, n), where 
            adjs[.][i][j] is the weight for edge j -> i.
        n: number of vertices.
    Returns:
        heads. Size (batch_size, n). heads[0] = 0 always.
    """
    # Convert roots and weights_and_edges to numpy arrays on the cpu.
    if torch.is_tensor(adjs):
        adjs = adjs.detach().to("cpu").numpy()

    # Loop over batch dimension to get the maximum spanning arborescence for
    # each graph.
    batch_size = adjs.shape[0]
    heads = np.zeros((batch_size, n))
    for sample_idx in range(batch_size):
        # We transpose adj because networkx accepts adjacency matrix
        # where adj[i][j] corresponds to edge i -> j.
        np.fill_diagonal(adjs[sample_idx], 0.0)
        # We multiply by -1.0 since networkx obtains the
        # minimum spanning arborescence. We want the maximum.
        G = nx.from_numpy_matrix(-1.0 * adjs[sample_idx].T, create_using=nx.DiGraph())
        
        Gcopy = G.copy()
        # Remove all incoming edges for the root such that
        # the given "root" is forced to be selected as the root.
        Gcopy.remove_edges_from(G.in_edges(nbunch=[0]))
        msa = nx.minimum_spanning_arborescence(Gcopy)
        
        # Convert msa nx graph to heads list.
        for i, j in msa.edges:
            i, j = int(i), int(j)
            heads[sample_idx][j] = i
            
    return heads


def edmonds_cpp_pytorch(adjs, n):
    """
    Gets the maximum spanning arborescence given weights of edges.
    We assume the root is node (idx) 0.
    Args:
        adjs: shape (batch_size, n, n), where 
            adjs[.][i][j] is the weight for edge j -> i.
        n: number of vertices.
    Returns:
        heads: Size (batch_size, n). 
            heads[i] = parent node of i; heads[0] = 0 always.

    """
    heads = edmonds_cpp.get_maximum_spanning_arborescence(adjs, n)
    return heads


if __name__ == "__main__":
    n = 10
    bs = 1000
    np.random.seed(42)
    adjs = np.random.rand(bs, n, n)

    res_nx = edmonds_python(adjs, n)
    res_cpp = edmonds_cpp_pytorch(torch.tensor(adjs), n).numpy()

    np.testing.assert_almost_equal(res_nx, res_cpp)
