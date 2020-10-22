import argparse
import time
import torch

from edmonds import edmonds_python, edmonds_cpp_pytorch

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4, help="Number of nodes.")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
parser.add_argument("--num_steps", type=int, default=1,
                    help="Number of times to evaluate.")
args = parser.parse_args()


adjs = torch.randn(args.batch_size, args.n, args.n)


# Test nx version.
t = 0
for _ in range(args.num_steps):
    start = time.time()
    res_nx = edmonds_python(adjs.numpy(), args.n)
    t += time.time() - start
print(f"Nx version took: {t}; avg: {t / args.num_steps}")

# C++ (cpu) version.
t = 0
for _ in range(args.num_steps):
    start = time.time()
    res_cpp_cpu = edmonds_cpp_pytorch(adjs, args.n)
    t += time.time() - start
print(f"C++ (cpu) version took: {t}; avg: {t / args.num_steps}")

# C++ (gpu) version.
t = 0
for _ in range(args.num_steps):
    start = time.time()
    res_cpp_gpu = edmonds_cpp_pytorch(adjs.to("cuda"), args.n)
    torch.cuda.synchronize()
    t += time.time() - start
print(f"C++ (gpu) version took: {t}; avg: {t / args.num_steps}")