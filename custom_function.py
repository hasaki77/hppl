import numpy as np
from numba import njit

def trapezoidal_function(f, xmin: int, xmax: float, n: int, **kwargs):
    integral = 0.0
    dx = (xmax - xmin) / (n+1)

    for x in np.linspace(xmin, xmax, n):
        integral += dx * ( f(x, **kwargs) + f(x+dx, **kwargs) )/2

    return integral

def func(x):
    return (1 - x**2)**0.5

def phonon_dos(omega, omega_c=10, C=1):
    return C * omega**2 * np.exp(-omega / omega_c)

def entropy(omega, T, kb, hbar):
    x = hbar*omega / (kb*T)
    return (x / np.tanh(x / 2) - np.log(2 * np.sinh(x / 2))) * phonon_dos(omega)

@njit
def jit_entropy(omega, T, kb, hbar):
    x = hbar*omega / (kb*T)
    return (x / np.tanh(x / 2) - np.log(2 * np.sinh(x / 2))) * 1 * omega**2 * np.exp(-omega / 10)

def phonon_dos(omega, omega_c=10, C=1):
    return C * omega**2 * np.exp(-omega / omega_c)

def enthalpy(omega, T, kb, hbar):
    x = hbar*omega / (kb*T)
    return hbar*omega * (1 / np.tanh(x / 2)) * phonon_dos(omega)

@njit
def jit_enthalpy(omega, T, kb, hbar):
    x = hbar*omega / (kb*T)
    return hbar*omega * (1 / np.tanh(x / 2)) * 1 * omega**2 * np.exp(-omega / 10)

def cpu_trapezoidal_function(f, xmin: int, xmax: float, n: int, **kwargs):
    integral = 0.0
    dx = (xmax - xmin) / (n+1)
    x = np.linspace(xmin, xmax, n)
    integral = dx * ( f(x, **kwargs) + f(x+dx, **kwargs) )/2
    return integral.sum()

def helmholtz_energy(omega, T, kb, hbar):
    x = hbar*omega / (kb*T)
    return np.log(2 * np.sinh(x / 2)) * 1 * omega**2 * np.exp(-omega / 10)
