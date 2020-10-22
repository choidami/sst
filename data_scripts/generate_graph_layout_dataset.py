import os
from functools import partial
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import argparse

import torch

from core.kruskals import kruskals_pytorch

parser = argparse.ArgumentParser()
parser.add_argument("--num_train", type=int, default=50000,
                    help="Number of training simulations to generate.")
parser.add_argument("--num_valid", type=int, default=10000,
                    help="Number of validation simulations to generate.")
parser.add_argument("--num_test", type=int, default=10000,
                    help="Number of test simulations to generate.")
parser.add_argument("--num_timesteps", type=int, default=10,
                    help="Number of timesteps for training data.")
parser.add_argument("--num_timesteps_test", type=int, default=None,
                    help="Number of timesteps for test data.")
parser.add_argument("--num_rounds", type=int, default=1,
                    help="Numer of consecutive graph layout algorithm steps.")
parser.add_argument("--num_vertices", type=int, default=10,
                    help="Number of vertices in the graph.")
parser.add_argument("--num_dims", type=int, default=2,
                    help="Number of dimensions for each data point.")
parser.add_argument("--noise_std", type=float, default=0.0,
                    help="Variance for noise to add to each data point.")
parser.add_argument("--num_timeskip", type=int, default=1,
                    help="The number of timesteps to skip initially.")
parser.add_argument("--threshold", type=float, default=1e-10,
                    help="Threshold for the networkx spring algorithm.")
parser.add_argument("--suffix", type=str, default="")            
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
args = parser.parse_args()


def get_edges_from_adj(adj):
    vertices = np.where(adj == 1.0)
    # Duplicate edges are fine for the downstream task purposes.
    edges = [e for e in zip(*vertices)]
    return edges


def generate_graph_layout_data(num_vertices=10, num_dims=2, num_samples=50000,
                               num_timesteps=10):
    n = num_vertices
    vertices = torch.triu_indices(n-1, n, offset=1)

    n = num_vertices
    edges_data_list = []
    loc_data_list = []
    for idx in range(num_samples):
        # First sample a tree from the uniform logits prior (logits = zeros).
        # Note that uniform logits isn't uniform over trees.
        gumbels = -torch.log(-torch.log(torch.rand((n * (n - 1)))))
        weights_and_edges = torch.Tensor(
            [list(e) for e in zip(gumbels, vertices[0], vertices[1])])
        # Sample is adjacency matrix with diagonals removed.
        sample = kruskals_pytorch(weights_and_edges, n)  # (n-1, n)

        # Add back diagonal for the edges data.
        flat_sample = sample[vertices[0], vertices[1]]
        edges = torch.zeros((n, n))
        edges[vertices[0], vertices[1]] = flat_sample
        edges[vertices[1], vertices[0]] = flat_sample
        edges_data_list.append(edges)

        # Create nx.Graph instance from sampled tree.
        edges_list = get_edges_from_adj(edges.numpy())
        G = nx.Graph()
        G.add_edges_from(edges_list)

        # Generate graph layout position starting from random node position.
        positions_list = []
        init_pos = np.random.randn(num_vertices, num_dims)
        pos = {i: init_pos[i] for i in range(num_vertices)}
        if args.num_timeskip > 0:
            pos = nx.spring_layout(G, pos=pos, iterations=args.num_timeskip, 
                                   threshold=args.threshold, 
                                   center=np.zeros(num_dims))
            positions_list.append([pos[i] for i in range(num_vertices)])
        else:
            positions_list.append(init_pos)

        for _ in range(0, num_timesteps - args.num_timeskip):
            pos = nx.spring_layout(G, pos=pos, iterations=args.num_rounds,
                                   threshold=args.threshold, 
                                   center=np.zeros(num_dims))
            pos = {i: (pos[i] + np.random.randn(num_dims) * args.noise_std)
                   for i in pos}
            positions_list.append([pos[i] for i in range(num_vertices)])
        
        loc_data = np.stack(positions_list)               # (timesteps, n, dim)
        loc_data = np.transpose(loc_data, axes=(0, 2, 1)) # (timesteps, dim, n)
        loc_data_list.append(loc_data)

    edges_data_list = torch.stack(edges_data_list).numpy()
    loc_data_list = np.stack(loc_data_list)
    return edges_data_list, loc_data_list


if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    fname = ((f"{args.noise_std}v_" if args.noise_std else "novar_") +
             (f"{args.num_timeskip}skip_" if args.num_timeskip > 0 else "") +
             f"{args.num_timesteps}t_{args.num_rounds}r")

    print("Generating {} training simulations".format(args.num_train))
    edges_train, data_train = generate_graph_layout_data(
        args.num_vertices, args.num_dims, args.num_train, args.num_timesteps)

    np.save(f"data/data_train_{args.suffix}{fname}_graph{args.num_vertices}.npy", data_train)
    np.save(f"data/edges_train_{args.suffix}{fname}_graph{args.num_vertices}.npy", edges_train)

    print("Generating {} validation simulations".format(args.num_valid))
    edges_valid, data_valid = generate_graph_layout_data(
        args.num_vertices, args.num_dims, args.num_valid, args.num_timesteps)

    np.save(f"data/data_valid_{args.suffix}{fname}_graph{args.num_vertices}.npy", data_valid)
    np.save(f"data/edges_valid_{args.suffix}{fname}_graph{args.num_vertices}.npy", edges_valid)
    
    print("Generating {} test simulations".format(args.num_test))
    if args.num_timesteps_test is None:
        args.num_timesteps_test = args.num_timesteps
    edges_test, data_test = generate_graph_layout_data(
        args.num_vertices, args.num_dims, args.num_test, args.num_timesteps_test)

    np.save(f"data/data_test_{args.suffix}{fname}_graph{args.num_vertices}.npy", data_test)
    np.save(f"data/edges_test_{args.suffix}{fname}_graph{args.num_vertices}.npy", edges_test)