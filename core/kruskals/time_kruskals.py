import argparse
import time
import torch

from kruskals import kruskals_pytorch, kruskals_pytorch_batched
from kruskals import kruskals_cpp_pytorch, kruskals_cpp_pytorch2

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=30, help="Number of nodes.")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size.")
parser.add_argument("--num_steps", type=int, default=1,
                    help="Number of times to evaluate.")
args = parser.parse_args()


num_edges = int(args.n * (args.n - 1) / 2)
weights = torch.randn(args.batch_size, num_edges)
vertices = torch.triu_indices(args.n - 1, args.n, offset=1)
tiled_vertices = vertices.transpose(0, 1).repeat((weights.size(0), 1, 1)).float()
weights_and_edges = torch.cat([weights.unsqueeze(-1), tiled_vertices], axis=-1)

# Test pytorch (batched, gpu).
t = 0
weights_and_edges = weights_and_edges.to("cuda")
for _ in range(args.num_steps):
    start = time.time()
    res_pytorch = kruskals_pytorch_batched(weights_and_edges, args.n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"Pytorch (batched, gpu): {t}; avg: {t / args.num_steps}")

# Test cpp (pytorch, cpu).
t = 0
weights_and_edges = weights_and_edges.to("cpu")
for _ in range(args.num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch(weights_and_edges, args.n)
    t += time.time() - start
print(f"C++ (pytorch, cpu): {t}; avg: {t / args.num_steps}")

# Test cpp (pytorch, gpu).
t = 0
weights_and_edges = weights_and_edges.to("cuda")
for _ in range(args.num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch(weights_and_edges, args.n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"C++ (pytorch, gpu): {t}; avg: {t / args.num_steps}")

# Test cpp (pytorch2, cpu).
t = 0
weights_and_edges = weights_and_edges.to("cpu")
for _ in range(args.num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch2(weights_and_edges, args.n)
    t += time.time() - start
print(f"C++ (pytorch2, cpu): {t}; avg: {t / args.num_steps}")

# Test cpp (pytorch2, gpu).
t = 0
weights_and_edges = weights_and_edges.to("cuda")
for _ in range(args.num_steps):
    start = time.time()
    res_pytorch = kruskals_cpp_pytorch2(weights_and_edges, args.n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"C++ (pytorch2, gpu): {t}; avg: {t / args.num_steps}")