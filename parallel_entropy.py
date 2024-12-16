
# Import Library
from mpi4py import MPI
import numpy as np
import argparse
import custom_functions as custom_f

# Constants
hbar = 1.054e-34
kB = 1.38e-23

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--xmin', type=float, help='Lower bound of the integral')
parser.add_argument('--xmax', type=float, help='Upper bound of the integral')
parser.add_argument('--n', type=int, help='Number of points in the integral')
parser.add_argument('--T', type=int, help='Temperature')
args = parser.parse_args()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chunk range
N = args.xmax - args.xmin
chunk_size = N // size

start = args.xmin + rank*chunk_size
end = start + chunk_size if rank != size - 1 else args.xmax

global_integral = None
if rank == 0:
    global_integral = np.array([0], dtype=np.float64)

# Main Calculations
entropy_value = kB * custom_f.trapezoidal_function(
    f=custom_f.entropy,
    xmin=start,
    xmax=end,
    n=args.n,
    T=args.T,
    kb=kB,
    hbar=hbar)

comm.Reduce(np.array(entropy_value), global_integral, op=MPI.SUM, root=0)

# Output results
if rank == 0:
    print(f"Entropy: {global_integral[0]}")
