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

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start_intg', type=int, help='Start point for integral')
parser.add_argument('--end_intg', type=int, help='End point for integral')
parser.add_argument('--n', type=int, help='Number of points')
args = parser.parse_args()

result = trapezoid_integral(func, args.start_intg, args.end_intg, args.n)

# Output results
if rank == 0:
    print(f'Integral result: {result:.3f}')
