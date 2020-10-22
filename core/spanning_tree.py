import time
from itertools import chain, combinations, permutations
import numpy as np

import torch
torch.set_printoptions(precision=32)

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from core.kruskals.kruskals import get_tree
from core.kruskals.kruskals import kruskals_pytorch_batched
from core.kruskals.kruskals import kruskals_cpp_pytorch

EPS = torch.finfo(torch.float32).tiny


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_edges_from_vertices(vertices, num_vertices):
    idx = 0
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if i in vertices and j in vertices:
                edges.append(idx)
            idx = idx + 1
    return edges


def submatrix_index(n, i):
    bs = i.size(0)
    I = torch.ones((bs, n, n), dtype=bool)
    I[torch.arange(bs), i, :] = False
    I[torch.arange(bs), :, i] = False
    return I


def get_spanning_tree_marginals(logits, n):
    bs = logits.size(0)
    (i, j) = torch.triu_indices(n, n, offset=1)
    c = torch.max(logits, axis=-1, keepdims=True)[0]
    k = torch.argmax(logits, axis=-1)
    removei = i[k]

    weights = torch.exp(logits - c)

    W = torch.zeros(weights.size(0), n, n)
    W = W.cuda() if logits.is_cuda else W
    W[:, i, j] = weights
    W[:, j, i] = weights

    L = torch.diag_embed(W.sum(axis=-1)) - W
    subL = L[submatrix_index(n, removei)].view(bs, n - 1, n - 1)
    logzs = torch.slogdet(subL)[1]
    logzs = torch.sum(logzs + (n - 1) * c.flatten())
    sample = torch.autograd.grad(logzs, logits, create_graph=True)[0]
    return sample


def clip_range(x, max_range=np.inf):
    m = torch.max(x, axis=-1, keepdim=True)[0]
    return torch.max(x, -1.0 * torch.tensor(max_range) * torch.ones_like(x) + m)


def sample_tree_from_logits(logits, tau=1.0, hard=False, hard_with_grad=False,
                            edge_types=1, relaxation="exp_family_entropy",
                            max_range=np.inf, use_cpp=False):
    """Samples a maximum spanning tree given logits.

    Args:
        logits: Logits of shape (batch_size, n * (n - 1), 1).
            They correspond to a flattened and transposed adjacency matrix
            with the diagonals removed.
            We assume the logits are edge-symmetric.
        tau: Float representing temperature.
        hard: Whether or not to sample hard edges.
        hard_with_grad: Whether or not to allow sample hard, but have gradients
            for backprop.
        edge_tpes: Number of edge types for the output. Must be 1 or 2.
        relaxation: Relaxation type.
        max_range: Maxiumum range between maximum edge weight and any other
            edge weights. Used for relaxation == "exp_family_entropy" only.
        use_cpp: Whether or not to use the C++ implementation of kruskal's
            algorithm for hard sampling.
    Returns:
        Sampled edges with the same shape as logits, and
        sampled edge weights of same shape as logits.
    """
    # n * (n - 1) = len(logits), where n is the number of vertices.
    n =  int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))

    # First check that there is only one edge type.
    assert logits.size(2) == 1
    # Reshape to adjacency matrix (with the diagonals removed).
    reshaped_logits = logits.view(-1, n, n - 1)
    reshaped_logits = reshaped_logits.transpose(1, 2)  # (bs, n-1, n)

    vertices = torch.triu_indices(n-1, n, offset=1)
    edge_logits = reshaped_logits[:, vertices[0], vertices[1]]

    uniforms = torch.empty_like(edge_logits).float().uniform_().clamp_(EPS, 1 - EPS)
    gumbels = uniforms.log().neg().log().neg()
    gumbels = gumbels.cuda() if logits.is_cuda else gumbels
    edge_weights = gumbels + edge_logits
    
    hard = True if hard_with_grad else hard
    if hard:
        tiled_vertices = vertices.transpose(0, 1).repeat((edge_weights.size(0), 1, 1)).float()
        tiled_vertices = tiled_vertices.cuda() if logits.is_cuda else tiled_vertices
        weights_and_edges = torch.cat([edge_weights.unsqueeze(-1), tiled_vertices], axis=-1)
        if use_cpp:
            samples = kruskals_cpp_pytorch(weights_and_edges.detach().cpu(), n)
            samples = samples.to("cuda") if logits.is_cuda else samples
        else:
            samples = kruskals_pytorch_batched(weights_and_edges, n)
        if edge_types == 2:
            null_edges = 1.0 - samples
            samples = torch.stack((null_edges, samples), dim=-1)
        else:
            samples = samples.unsqueeze(-1)
        hard_samples = samples
    if not hard or hard_with_grad:
        if relaxation == 'exp_family_entropy':
            weights = edge_weights / tau
            weights = clip_range(weights, max_range)
            X = get_spanning_tree_marginals(weights, n)
        elif relaxation == "binary_entropy":  # Soft sample using CVXPY.
            # Very slow!
            # Define the DPP problem.
            x = cp.Variable(edge_weights.size(1))
            y = cp.Parameter(edge_weights.size(1))
            obj = -x @ y - (cp.sum(cp.entr(x)) + cp.sum(cp.entr(1.0 - x)))

            subsets_of_vertices = [torch.IntTensor(l) for l in powerset(torch.arange(n)) 
                                   if (len(l) >= 2  and len(l) < n)]
            edges_list = [get_edges_from_vertices(s, n) for s in subsets_of_vertices]
            cons = [cp.sum(x) == (n - 1.0), x >= 0.0]
            for i in range(len(edges_list)):
                cons.append(cp.sum(x[edges_list[i]]) <= (len(subsets_of_vertices[i]) - 1.0))
            prob = cp.Problem(cp.Minimize(obj), cons)

            layer = CvxpyLayer(prob, parameters=[y], variables=[x])
            X, = layer(edge_weights / tau)
        else:
            raise ValueError("Invalid relaxation for spanning tree.")

        samples = torch.zeros_like(reshaped_logits)
        samples[:, vertices[0], vertices[1]] = X
        samples[:, vertices[1] - 1, vertices[0]] = X

        if edge_types == 2:
            samples = torch.stack((1.0 - samples, samples), dim=-1)
        else:
            samples = samples.unsqueeze(-1)

    if hard_with_grad:
        samples = (hard_samples - samples).detach() + samples

    # Return the flattened sample in the same format as the input logits.
    samples = samples.transpose(1, 2).contiguous().view(-1, n * (n - 1), edge_types)

    # Make sampled edge weights into adj matrix format.
    edge_weights_reshaped = torch.zeros_like(reshaped_logits)
    edge_weights_reshaped[:, vertices[0], vertices[1]] = edge_weights
    edge_weights_reshaped[:, vertices[1] - 1, vertices[0]] = edge_weights
    edge_weights = edge_weights_reshaped.transpose(1, 2).contiguous().view(logits.shape)

    return samples, edge_weights


def enumerate_spanning_trees(weights_and_edges, n):
    """
    Args:
        weights_and_edges: Shape (n * (n - 2), 3).
        n: Number of vertices.
    """
    probs = {}
    for edgeperm in permutations(weights_and_edges):
        edgeperm = torch.stack(edgeperm)
        tree = get_tree(edgeperm[:, 1:].int(), n)
        weights = edgeperm[:, 0]
        logprob = 0
        for i in range(len(weights)):
            logprob += weights[i] -  torch.logsumexp(weights[i:], dim=0)
        tree_str = "".join([str(x) for x in tree.flatten().int().numpy()])
        if tree_str in probs:
            probs[tree_str] = probs[tree_str] + torch.exp(logprob)
        else:
            probs[tree_str] = torch.exp(logprob)
    return probs


def compute_probs_for_tree(logits, use_gumbels=True):
    if use_gumbels:
        return logits
    # n * (n - 1) = len(logits), where n is the number of vertices.
    n =  int(0.5 * (1 + np.sqrt(4 * logits.size(1) + 1)))

    # Reshape to adjacency matrix (with the diagonals removed).
    reshaped_logits = logits.view(-1, n, n - 1)
    reshaped_logits = reshaped_logits.transpose(1, 2)  # (bs, n-1, n)
    # Get the edge logits (upper triangle).
    vertices = torch.triu_indices(n-1, n, offset=1)
    edge_logits = reshaped_logits[:, vertices[0], vertices[1]]

    probs = []
    for weights in edge_logits:
      weights_and_edges = torch.Tensor(
          [list(e) for e in zip(weights, vertices[0], vertices[1])])
      p_dict = enumerate_spanning_trees(weights_and_edges, n)
      p = torch.tensor(list(p_dict.values()))
      probs.append(p)
    probs = torch.stack(probs)
    return probs


if __name__ == "__main__":
    ##################### Testing compute_probs_for_tree #####################
    bs = 1
    n = 4

    logits = torch.rand((bs, n * (n-1)))
    prob = compute_probs_for_tree(logits, use_gumbels=False)
    np.testing.assert_almost_equal(prob.sum(axis=-1).numpy(), np.ones((bs,)))
