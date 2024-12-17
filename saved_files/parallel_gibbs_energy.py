
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
parser.add_argument('--tmin', type=int, help='Lower bound of the temperature')
parser.add_argument('--tmax', type=int, help='Upper bound of the temperature')
parser.add_argument('--step', type=int, help='Temperature step')
parser.add_argument('--xmin', type=float, help='Lower bound of the integral')
parser.add_argument('--xmax', type=float, help='Upper bound of the integral')
parser.add_argument('--n', type=int, help='Number of points in the integral')
args = parser.parse_args()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

temp_range = np.arange(args.tmin, args.tmax, args.step)
local_values = np.zeros_like(temp_range, dtype=np.float64)
global_values = None
if rank == 0:
    global_values = np.zeros_like(temp_range, dtype=np.float64)

# Main Calculations
for i, T in enumerate(temp_range):
    if rank == 0:
        local_values[i] = - T * kB * custom_f.trapezoidal_function(
            f=custom_f.jit_entropy,
            xmin=args.xmin,
            xmax=args.xmax,
            n=args.n,
            T=T,
            kb=kB,
            hbar=hbar)
    else:
        local_values[i] = custom_f.trapezoidal_function(
            f=custom_f.jit_enthalpy,
            xmin=args.xmin,
            xmax=args.xmax,
            n=args.n,
            T=T,
            kb=kB,
            hbar=hbar
            )

comm.Reduce(local_values, global_values, op=MPI.SUM, root=0)

# Output results
# if rank == 0:
#     print(f"Gibbs Free Energy: {global_values}")
