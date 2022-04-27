import numpy as np
import matplotlib.pyplot as plt
from all_ode import lokta_volterra

def euler_step(t, X, func, args, h):

    return np.array([X + h * func(t, X, *args), t + h])

def rk4_step(t, X, func, args, h):

    k1 = func(t, X, *args)
    k2 = func(t+h/2, X+(h*k1)/2, args)
    k3 = func(t+h/2, X+(h*k2)/2, args)
    k4 = func(t+h, X+(h*k3), args)
    k = (h * (k1+2*k2+2*k3+k4))/6

    return  np.array([X + k, t + h])

def solve_to(func, x0, args, t0, tend, deltat_max, method):

    t = t0
    x = x0

    while t < tend:
        if t + deltat_max > tend:
            deltat_max = tend - t
        x, t = method(t, x, func, args, deltat_max)
    return x, t

def solve_ode(func, x0, args, tspan, deltat_max, method):

    solution = np.zeros([len(tspan),len(x0)])
    t0 = tspan[0]

    for i, tend in enumerate(tspan):
        sol = solve_to(func, x0, args, t0, tend, deltat_max, method)
        solution[i,:] = sol[0]
        x0 = sol[0]
        t0 = sol[1]

    return solution

def plot_solution(tspan, solution):

    for (index, sol) in enumerate(solution.T):
	    plt.plot(tspan, sol, label='x{}'.format(index+1))
    
    plt.ylabel('Value')
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()

def main():

    func = lokta_volterra
    u0 = np.array((0.5, 1, 100))
    args = np.array((1, 0.2, 0.1))
    u0 = u0[:-1]
    deltat_max = 0.01
    tspan = np.linspace(0,50,50)
    method = euler_step
    sol = solve_ode(func, u0, args, tspan, deltat_max, method)
    print(sol)
    plot_solution(tspan, sol)

if __name__ == "__main__":

    main()
