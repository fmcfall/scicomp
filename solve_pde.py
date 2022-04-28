from re import X
import numpy as np
import matplotlib.pyplot as plt
import time
from all_pde import *

def component_forward_euler(mx, u_j, lmbda):

    u_jp1 = np.zeros(mx+1) 
    for i in range(1, mx):
        u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
    
    return u_jp1

def matrix_forward_euler(mx, u_j, lmbda):

    u_jp1 = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(u_jp1.shape)
    u_jp1[rows==cols] = 1 - 2*lmbda
    u_jp1[rows==cols+1] = lmbda
    u_jp1[rows==cols-1] = lmbda
    u_jp1 = np.dot(u_jp1, u_j[1:-1])

    return np.concatenate(([0], u_jp1, [0]))

def TDMA(a,b,c,d):

    n = len(a)
    a, b, c, d = map(np.array, (a, b, c, d))

    for j in range(1,n):
        a[j] = a[j] / b[j-1]
        b[j] = b[j] - a[j] * c[j-1]

        d[j] = d[j] - a[j] * d[j-1]
    
    d[n-1] = d[n-1] / b[n-1]

    for j in range(n-1, -1, -1):
        d[j] = (d[j] - c[j] * d[j+1]) / b[j]

    return d

def matrix_backward_euler(mx, u_j, lmbda):

    u_jp1 = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(u_jp1.shape)
    u_jp1[rows==cols] = 1 + 2*lmbda
    u_jp1[rows==cols+1] = -lmbda
    u_jp1[rows==cols-1] = -lmbda
    u_jp1 = np.linalg.solve(u_jp1, u_j[1:-1])

    return np.concatenate(([0], u_jp1, [0]))

def solve_pde(pde, x, mx, mt, L, lmbda, method):

    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step

    # Set the initial conditions.
    for i in range(0, mx+1):
        u_j[i] = pde(x[i], L)

    # Solve the PDE: loop over all time points.
    for j in range(0, mt):
        # PDE discretised at position x[i], time t[j]
        u_jp1 = method(mx, u_j, lmbda)
        
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
    print(str(method.__name__)+' time: {}'.format(end-start))

def main():

    kappa = 1
    L = 1
    T = 0.5
    mx = 10
    mt = 1000
    x = np.linspace(0, L, mx+1)
    t = np.linspace(0, T, mt+1)
    deltax = x[1] - x[0]
    deltat = t[1] - t[0] 
    lmbda = kappa*deltat/(deltax**2)
    method = matrix_backward_euler
    pde = uI
    exact_pde = uExact

    stability(pde, exact_pde, x, mx, mt, L, T, kappa, lmbda, method)

if __name__ == '__main__':

    main()