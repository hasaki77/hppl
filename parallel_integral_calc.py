from mpi4py import MPI
import numpy as np
import argparse

# Functions
def trapezoid_integral(f, x_min: int, x_max: int, n: int) -> float:
    integral = 0.0
    delta_x = (x_max - x_min) / (n-1)
    
    for x in np.linspace(x_min, x_max, n):
        integral += delta_x * (f(x) + f(x+delta_x)) / 2

    return integral

def func(x):
    return x**2

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start_intg', type=int, help='Start point for integral')
parser.add_argument('--end_intg', type=int, help='End point for integral')
parser.add_argument('--n', type=int, help='Number of points')
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Chunk range
N = np.abs(args.end_intg) - np.abs(args.start_intg)
chunk_size = N // size

start = args.start_intg + rank * chunk_size
end = start + chunk_size if rank != size - 1 else args.end_intg

local_result = trapezoid_integral(func, start, end, n=args.n//size)

global_sum = None
if rank == 0:
    global_sum = np.array([0], dtype=np.float64)


comm.Reduce(np.array([local_result]), global_sum, op=MPI.SUM, root=0)
#MPI.Finalize()

# Output results
if rank == 0:
    print(f"Интеграл | x**2 | от {args.start_intg} до {args.end_intg}: {global_sum[0]:.3f}")
