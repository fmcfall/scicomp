import numpy as np

class uI:
    def func(x, L):
        # Initial temperature distribution
        y = np.sin(np.pi*x/L)
        return y
    def exact(x, L, t, kappa):
        # The exact solution
        y = np.exp(-kappa*(np.pi**2/L**2)*t)*np.sin(np.pi*x/L)
        return y