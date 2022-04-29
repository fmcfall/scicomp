from re import X
import numpy as np
import matplotlib.pyplot as plt
import time
from all_pde import *

def component_forward_euler(mx, u_j, lmbda):
    """
    Function that executes a forward euler step to find the next solution values, using
    a component-wise approach.

    Parameters
    ----------
    mx:	int
        Number of gridpoints in space.

	u_j:	np.array(float)
		Current solution values.

    lmbda:	float
    	Mesh fourier number.

    Returns
    -------
	u_jp1: np.array(float)
	 	New solution values.
    """
    u_jp1 = np.zeros(mx+1) 
    for i in range(1, mx):
        u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
    
    return u_jp1

def dirichlet(j, lmbda, u_jp1, *args):
    """
    Function that executes a forward euler step to find the next solution values, using
    a component-wise approach.

    Parameters
    ----------
    j:   int
        Current index.

    lmbda:	float
    	Mesh fourier number.

	u_jp1: np.array(float)
	 	Solution values to be altered.

    args:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    Returns
    -------
	u_jp1: np.array(float)
	 	New solution values.
    """
    p, q = args
    u_jp1[0] = p(j)
    u_jp1[0] += lmbda * p(j)
    u_jp1[-1] = q(j)
    u_jp1[-1] += lmbda * q(j)

    return u_jp1

def matrix_forward_euler(j, mx, u_j, lmbda, dirichlet_conditions):
    """
    Function that executes a forward euler step to find the next solution values, using
    a matrix approach.

    Parameters
    ----------
    j:   int
        Current index.

    mx:	int
        Number of gridpoints in space.

	u_j:	np.array(float)
		Current solution values.

    lmbda:	float
    	Mesh fourier number.

    dirichlet_conditions:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    Returns
    -------
	u_jp1: np.array(float)
	 	New solution values.
    """
    u_jp1 = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(u_jp1.shape)
    u_jp1[rows==cols] = 1 - 2*lmbda
    u_jp1[rows==cols+1] = lmbda
    u_jp1[rows==cols-1] = lmbda
    u_jp1 = np.dot(u_jp1, u_j[1:-1])
    if len(dirichlet_conditions) == 2:
        u_jp1 = dirichlet(j, mx, lmbda, u_jp1, dirichlet_conditions)

    return np.concatenate(([0], u_jp1, [0]))

def matrix_backward_euler(j, mx, u_j, lmbda, dirichlet_conditions):
    """
    Function that executes a backward euler step to find the next solution values, using
    a matrix approach.

    Parameters
    ----------
    j:   int
        Current index.
    
    mx:	int
        Number of gridpoints in space.

	u_j:	np.array(float)
		Current solution values.

    lmbda:	float
    	Mesh fourier number.

    dirichlet_conditions:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    Returns
    -------
	u_jp1: np.array(float)
	 	New solution values.
    """
    u_jp1 = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(u_jp1.shape)
    u_jp1[rows==cols] = 1 + 2*lmbda
    u_jp1[rows==cols+1] = -lmbda
    u_jp1[rows==cols-1] = -lmbda
    u_jp1 = np.linalg.solve(u_jp1, u_j[1:-1])
    if len(dirichlet_conditions) == 2:
        u_jp1 = dirichlet(j, mx, lmbda, u_jp1, dirichlet_conditions)

    return np.concatenate(([0], u_jp1, [0]))

def crank_nicholson(j, mx, u_j, lmbda, dirichlet_conditions):
    """
    Function that executes a crank nicholson step to find the next solution values, using
    a matrix approach.

    Parameters
    ----------
    j:   int
        Current index. 
  
    mx:	int
        Number of gridpoints in space.

	u_j:	np.array(float)
		Current solution values.

    lmbda:	float
    	Mesh fourier number.

    dirichlet_conditions:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    Returns
    -------
	u_jp1: np.array(float)
	 	New solution values.
    """
    A = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(A.shape)
    A[rows==cols] = 1 + lmbda
    A[rows==cols+1] = -lmbda/2
    A[rows==cols-1] = -lmbda/2  
    B = np.zeros((mx-1, mx-1))
    rows, cols = np.indices(B.shape)
    B[rows==cols] = 1 - lmbda
    B[rows==cols+1] = lmbda/2
    B[rows==cols-1] = lmbda/2
    u_j = np.dot(u_j[1:-1], B)
    u_jp1 = np.linalg.solve(A, u_j)
    if len(dirichlet_conditions) == 2:
        u_jp1 = dirichlet(j, mx, lmbda, u_jp1, dirichlet_conditions)

    return  np.concatenate(([0], u_jp1, [0]))

def solve_pde(pde, x, mx, mt, L, lmbda, method, dirichlet_conditions):
    """
    Function that solves a pde, given a solving method.

    Parameters
    ----------    
    pde:    function
        PDE function, must return a numpy array.

    x:     numpy.array(float)
        Array of x values to solve for.

    mx:	int
        Number of gridpoints in space.

    mx:	int
        Number of gridpoints in time.

    L:  float
        Upper boundary value of x.

    lmbda:	float
    	Mesh fourier number.

    method:     function
        PDE solving method.

    dirichlet_conditions:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    Returns
    -------
	u_j: np.array(float)
	 	Final solution.
    """
    # Set up the solution variables
    u_j = np.zeros(x.size)        # u at current time step

    # Set the initial conditions.
    for i in range(0, mx+1):
        u_j[i] = pde(x[i], L)

    # Solve the PDE: loop over all time points.
    for j in range(0, mt):
        # PDE discretised at position x[i], time t[j]
        u_jp1 = method(j, mx, u_j, lmbda, dirichlet_conditions)
        
        # Boundary conditions
        u_jp1[0] = 0; u_jp1[mx] = 0
        
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return u_j

def plot_solution(x, u_j, exact_pde, L, T, kappa):
    """
    Function that plots the computed solution values against the exact solution.

    Parameters
    ----------
    x:     numpy.array(float)
        Array of x values to solve for.

	u_j:	np.array(float)
		Current solution values.

    exact_pde:   function
        Function for the exact solution to the equivalent PDE.

    L:  float
        Upper boundary value of x.

    T:  float
        Upeer value of t.

    kappa:  float
        Heat diffusivity parameter.
    """
    plt.plot(x, u_j, 'ro', label='num')
    xx = np.linspace(0, L, 250)
    plt.plot(xx, exact_pde(xx, L, T, kappa), 'b-', label='exact')
    plt.xlabel('x')
    plt.ylabel('u_j')
    plt.legend(loc='upper right')
    plt.show()

def stability(pde, exact_pde, x, mx, mt, L, T, kappa, lmbda, methods, dirichlet_conditions):
    """
    Function that plots the computed solution values against the exact solution for
    each method in a list of methods and computes the time taken for each method.

    Parameters
    ----------
    x:     numpy.array(float)
        Array of x values to solve for.

	u_j:	np.array(float)
		Current solution values.

    exact_pde:   function
        Function for the exact solution to the equivalent PDE.

    L:  float
        Upper boundary value of x.

    T:  float
        Upeer value of t.

    kappa:  float
        Heat diffusivity parameter.

    lmbda:	float
    	Mesh fourier number.

    methods:    list
        List of methods to compare.

    dirichlet_conditions:   np.array(function)
        Functions for the dirichlet boundary condition. If no
        condition, leave array empty.

    """
    for m in methods:
        start = time.time()
        u_j = solve_pde(pde, x, mx, mt, L, lmbda, m, dirichlet_conditions)
        end = time.time()
        plot_solution(x, u_j, exact_pde, L, T, kappa)
        print(str(m.__name__)+' time: {}'.format(end-start), ', sol:', u_j)

def main():

    kappa = 1
    L = 1
    T = 0.5
    mx = 35
    mt = 1000
    x = np.linspace(0, L, mx+1)
    t = np.linspace(0, T, mt+1)
    deltax = x[1] - x[0]
    deltat = t[1] - t[0] 
    lmbda = kappa*deltat/(deltax**2)
    methods = [matrix_forward_euler, matrix_backward_euler, crank_nicholson]
    pde = uI
    exact_pde = uExact
    dirichlet_conditions = np.array([])
    #plot_solution(x, sol, exact_pde, L, T, kappa)
    stability(pde, exact_pde, x, mx, mt, L, T, kappa, lmbda, methods, dirichlet_conditions)

if __name__ == '__main__':

    main()