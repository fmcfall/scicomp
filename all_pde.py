import numpy as np

def uI(x, L):
    # Initial temperature distribution
    y = np.sin(np.pi*x/L)
    return y

def uExact(x, t, kappa, L):
    # The exact solution
    y = np.exp(-kappa*(np.pi**2/L**2)*t)*np.sin(np.pi*x/L)
    return y