import numpy as np
import matplotlib.pyplot as plt
import time
from all_ode import simple, simple_exact

def euler_step(t, X, func, args, h):

    return np.array([X + h * func(t, X, *args), t + h])

def rk4_step(t, X, func, args, h):

    k1 = func(t, X, *args)
    k2 = func(t+h/2, X+(h*k1)/2, *args)
    k3 = func(t+h/2, X+(h*k2)/2, *args)
    k4 = func(t+h, X+(h*k3), *args)
    k = (h * (k1+2*k2+2*k3+k4))/6

    return  np.array([X + k, t + h])

def solve_to(func, x0, args, t0, tend, deltat_max, method):

    t = t0
    x = x0

    while t < tend:
        if tend < t + deltat_max:
            deltat_max = tend - t
        x, t = method(t, x, func, args, deltat_max)
        
    return x, t

def solve_ode(func, x0, args, tspan, deltat_max, method):

    solution = [[] for x in x0]
    t0 = tspan[0]

    for tend in tspan:
        sol = solve_to(func, x0, args, t0, tend, deltat_max, method)
        for i, x in enumerate(sol[0]):
            solution[i].append(x)
        x0 = sol[0]
        t0 = sol[1]

    return solution

def plot_solution(tspan, solution):
    
    for i, sol in enumerate(solution):
        plt.plot(tspan, sol, label='x{}'.format(i+1))
    
    plt.ylabel('x')
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()

def get_error(tspan, solution, exact_func, exact_args):

    exact_solution = np.array([exact_func(tspan, exact_args) for t in tspan])
    abs_error = np.absolute(np.subtract(solution, exact_solution))

    return np.sum(abs_error)

def get_error_list(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, method):

    delta_t_list = np.logspace(deltat_min, deltat_max, deltat_num)
    error_list = []
    for delta_t in delta_t_list:
        sol = solve_ode(func, x0, args, tspan, delta_t, method)
        error_list.append(get_error(tspan, sol, exact_func, exact_args))

    return error_list, delta_t_list

def plot_errors(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, methods):

    for m in methods:
        method = m
        error_list, deltat_list = get_error_list(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, method)
        plt.loglog(deltat_list, error_list, label=method.__name__)

    plt.ylabel('Error')
    plt.xlabel('Step size')
    plt.legend()
    plt.show()

def get_time(func, x0, args, tspan, deltat_max, method):

    start = time.time()
    solve_ode(func, x0, args, tspan, deltat_max, method)
    end = time.time()

    return end - start

def main():

    func = simple
    x0 = np.array([1])
    args = np.array(())
    exact_func = simple_exact
    exact_args = np.array((1))
    deltat_max = 0.01
    tspan = np.linspace(0,2,20)
    methods = [euler_step, rk4_step]
    method = rk4_step
    sol = solve_ode(func, x0, args, tspan, deltat_max, method)
    time = get_time(func, x0, args, tspan, deltat_max, method)
    print(time)
    plot_solution(tspan, sol)
    plot_errors(tspan, func, x0, args, exact_func, exact_args, -5, -1, 20, methods)

if __name__ == "__main__":

    main()
