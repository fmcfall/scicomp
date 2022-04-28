import numpy as np
import matplotlib.pyplot as plt
from all_pde import *

def solve_pde(pde, x, mx, mt, L, lmbda):

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set the initial conditions.
    for i in range(0, mx+1):
        u_j[i] = pde(x[i], L)

    # u at next time step.
    u_jp1 = np.zeros(mx+1)
    # Solve the PDE: loop over all time points.
    for j in range(0, mt):
        # Forward Euler timestep at inner mesh points
        # PDE discretised at position x[i], time t[j]
        for i in range(1, mx):
            u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
        
    # Boundary conditions
    u_jp1[0] = 0; u_jp1[mx] = 0
        
    # Save u_j at time t[j+1]
    u_j[:] = u_jp1[:]

    return u_j

kappa = 1
L = 1
T = 0.5
mx = 20
mt = 1000
x = np.linspace(0, L, mx+1)
t = np.linspace(0, T, mt+1)
deltax = x[1] - x[0]
deltat = t[1] - t[0] 
lmbda = kappa*deltat/(deltax**2)
p = solve_pde(uI, x, mx, mt, L, lmbda)
print(p)