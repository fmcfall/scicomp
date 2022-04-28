import numpy as np
import matplotlib.pyplot as plt
import time
from all_pde import *

def component_forward_euler(mx, u_j, u_jp1, lmbda):

    for i in range(1, mx):
        u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
    
    return u_jp1



def solve_pde(pde, x, mx, mt, L, lmbda, method):

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step
    u_jp1 = np.zeros(x.size)      # u at next time step

    # Set the initial conditions.
    for i in range(0, mx+1):
        u_j[i] = pde(x[i], L)

    # Solve the PDE: loop over all time points.
    for j in range(0, mt):
        # PDE discretised at position x[i], time t[j]
        u_jp1 = method(mx, u_j, u_jp1, lmbda)
        
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
        
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j

def plot_solution(x, u_j, exact_pde, L, T, kappa):

    plt.plot(x, u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    plt.plot(xx, exact_pde(xx, L, T, kappa), 'b-', label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x,0.5)')
    plt.legend(loc='upper right')
    plt.show()

def stability(pde, exact_pde, x, mx, mt, L, T, kappa, lmbda, method):

    start = time.time()
    u_j = solve_pde(pde, x, mx, mt, L, lmbda, method)
    plot_solution(x, u_j, exact_pde, L, T, kappa)
    end = time.time()
    print(str(method.__name__)+' Time: {}'.format(end-start))

def main():
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
    method = component_forward_euler
    pde = uI
    exact_pde = uExact
    
    stability(pde, exact_pde, x, mx, mt, L, T, kappa, lmbda, method)

if __name__ == '__main__':

    main()