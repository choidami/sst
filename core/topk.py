import torch
from torch.autograd import Function

import numpy as np
import numpy.random as npr
import scipy.special as spec

EPS = torch.finfo(torch.float32).tiny
INF = np.finfo(np.float32).max


def softtopk_forward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, 0, 0] = 0
    messages[:, 0, 1] = logits[:, 0]
    for i in range(1, n):
        for j in range(k + 1):
            logp_dont_use = messages[:, i - 1, j]
            logp_use = (
                messages[:, i - 1, j - 1] + logits[:, i] if j > 0 else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages


def softtopk_backward_np(logits, k):
    batchsize, n = logits.shape
    messages = -INF * np.ones((batchsize, n, k + 1))
    messages[:, n - 1, k] = 0
    for i in range(n - 2, -1, -1):
        for j in range(k + 1):
            logp_dont_use = messages[:, i + 1, j]
            logp_use = (
                messages[:, i + 1, j + 1] + logits[:, i + 1] if j < k else -INF)
            message = np.logaddexp(logp_dont_use, logp_use)
            messages[:, i, j] = message
    return messages


def softtopk_np(logits, k):
    batchsize = logits.shape[0]
    f = softtopk_forward_np(logits, k)
    b = softtopk_backward_np(logits, k)
    initial_f = -INF * np.ones((batchsize, 1, k + 1))
    initial_f[:, :, 0] = 0
    ff = np.concatenate([initial_f, f[:, :-1, :]], axis=1)
    lse0 = spec.logsumexp(ff + b, axis=2)
    lse1 = spec.logsumexp(ff[:, :, :-1] + b[:, :, 1:], axis=2) + logits
    return np.exp(lse1 - np.logaddexp(lse0, lse1))


class SoftTopK(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, k, eps):
        # ctx is a context object that can be used to stash information
        # for backward computation.
        ctx.save_for_backward(logits)
        ctx.k = k
        ctx.eps = eps
        dtype = logits.dtype
        device = logits.device
        mu_np = softtopk_np(logits.cpu().detach().numpy(), k)
        mu = torch.from_numpy(mu_np).type(dtype).to(device)
        return mu

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        r"""http://www.cs.toronto.edu/~kswersky/wp-content/uploads/carbm.pdf"""
        logits, = ctx.saved_tensors
        k = ctx.k
        eps= ctx.eps
        dtype = grad_output.dtype
        device = grad_output.device
        logits_np = logits.cpu().detach().numpy()
        grad_output_np = grad_output.cpu().detach().numpy()
        n1 = softtopk_np(logits_np + eps * grad_output_np, k)
        n2 = softtopk_np(logits_np - eps * grad_output_np, k)
        grad_np = (n1 - n2) / (2 * eps)
        grad = torch.from_numpy(grad_np).type(dtype).to(device)
        return grad, None, None


def sample_topk_from_logits(logits, k, tau=1.0, hard=False, 
                            hard_with_grad=False, edge_types=1,
                            relaxation="exp_family_entropy", 
                            eps=1e-2):
    """Does k-subset selection given logits.

    Args:
        logits: Logits of shape (batch size, n * (n - 1), 1).
            They correspond to a flattened and transposed adjacency matrix
            with the diagonals removed.
            We assume the logits are edge-symmetric.
        k: Subset selection size.
        tau: Float representing temperature.
        hard: Whether or not to sample hard edges.
        hard_with_grad: Whether or not to allow sample hard, but have gradients
            for backprop.
        edge_tpes: Number of edge types for the output. Must be 1 or 2.
        relaxation: Relaxation type.
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
        _, topk_indices = torch.topk(edge_weights, k, dim=-1)
        X = torch.zeros_like(edge_logits).scatter(-1, topk_indices, 1.0)
        hard_X = X
    if not hard or hard_with_grad:
        weights = edge_weights / tau
        if relaxation == "exp_family_entropy":
            X = SoftTopK.apply(weights, k, eps)
        elif relaxation == "binary_entropy":  
            # Limited Multi-Label Projection Layer (Amos et al.).
            raise ValueError("Binary entropy for topk not implemented.")
        else:
            raise ValueError("Invalid relaxation for topk.")

    if hard_with_grad:
        X = (hard_X - X).detach() + X

    samples = torch.zeros_like(reshaped_logits)
    samples[:, vertices[0], vertices[1]] = X
    samples[:, vertices[1] - 1, vertices[0]] = X

    if edge_types == 2:
        samples = torch.stack((1.0 - samples, samples), dim=-1)
    else:
        samples = samples.unsqueeze(-1)
        
    # Return the flattened sample in the same format as the input logits.
    samples = samples.transpose(1, 2).contiguous().view(-1, n * (n - 1), edge_types)

    # Make sampled edge weights into adj matrix format.
    edge_weights_reshaped = torch.zeros_like(reshaped_logits)
    edge_weights_reshaped[:, vertices[0], vertices[1]] = edge_weights
    edge_weights_reshaped[:, vertices[1] - 1, vertices[0]] = edge_weights
    edge_weights = edge_weights_reshaped.transpose(1, 2).contiguous().view(logits.shape)

    return samples, edge_weights
