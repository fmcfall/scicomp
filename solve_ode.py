import numpy as np
import matplotlib.pyplot as plt
import time
from all_ode import simple, simple_exact

def euler_step(t, X, func, args, h):
    """
    Function that completes one iteration of the Euler method.

    Parameters
    ----------
    t:	float
    	Current t value.

    X: 	numpy.array(float)
        Current position values.

	func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.

	h:	float
		Step size of the iteration.

    Returns
    -------
    X:	float
    	New position values.

    t: 	float
    	New time value.
    """
    return np.array([X + h * func(t, X, *args), t + h], dtype=object)

def rk4_step(t, X, func, args, h):
    """
    Function that completes one iteration of the 4th order Runga-Kutta method.

    Parameters
    ----------
    t:	float
    	Current t value.

    X: 	numpy.array(float)
        Current position values.

	func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.

	h:	float
		Step size of the iteration.

    Returns
    -------
    X:	float
    	New position values.

    t: 	float
    	New time value.
    """
    k1 = func(t, X, *args)
    k2 = func(t+h/2, X+(h*k1)/2, *args)
    k3 = func(t+h/2, X+(h*k2)/2, *args)
    k4 = func(t+h, X+(h*k3), *args)
    k = (h * (k1+2*k2+2*k3+k4))/6

    return  np.array([X + k, t + h], dtype=object)

def rk5_step(t, X, func, args, h):
    """
    Function that completes one iteration of the 5th order Runga-Kutta method.

    Parameters
    ----------
    t:	float
    	Current t value.

    X: 	numpy.array(float)
        Current position values.

	func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.

	h:	float
		Step size of the iteration.

    Returns
    -------
    X:	float
    	New position values.

    t: 	float
    	New time value.
    """
    k1 = h * func(t, X)
    k2 = h * func(t + (1/4) * h, X + (1/4) * k1, *args)
    k3 = h * func(t + (3/8) * h, X + (3/32) * k1 + (9/32) * k2, *args)
    k4 = h * func(t + (12/13) * h, X + (1932/2197) * k1 + (-7200/2197) * k2 + (7296/2197) * k3, *args)
    k5 = h * func(t + (1) * h, X + (439/216) * k1 + (-8) * k2 + (3680/513) * k3 + (-845/4104) * k4, *args)
    k6 = h * func(t + (1/2) * h, X + (-8/27) * k1 + (2) * k2 + (-3544/2565) * k3 + (1859/4104) * k4 + (-11/40) * k5, *args)   

    Xout = X + k1 * (16/135) + k3 * (6656/12825) + k4 * (28561/56430) - k5 * (9/50) + k6 * (2/55)
    return np.array([Xout, t + h], dtype=object)

def solve_to(func, x0, args, t0, tend, deltat_max, method):
    """
    Function that completes one solves from (x1,t1) to (x2,t2) using any desired
    method.

    Parameters
    ----------
    func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    x0: 	numpy.array(float)
        Position value to solve from.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.   

    t0:     float
    	First time value to solve from.

    tend:   float
        Time value to solve to.

	deltat_max:	float
		Maximum step size.

    method:     function
        Solving method. This can be a method defined above
        or any method that returns a position and time value,
        within a numpy array.

    Returns
    -------
    x:	float
    	New position value.

    t: 	float
    	New time value.
    """
    t = t0
    x = x0

    while t < tend:
        if tend < t + deltat_max:
            deltat_max = tend - t
        x, t = method(t, x, func, args, deltat_max)
        
    return x, t

def solve_ode(func, x0, args, tspan, deltat_max, method):
    """
    Function that generates series of numerical solution estimates from
    an initial position value, for a series of time values.

    Parameters
    ----------
    func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    x0: 	numpy.array(float)
        Position value to solve from.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.   

    t0:     float
    	First time value to solve from.

    tend:   float
        Time value to solve to.

	deltat_max:	float
		Maximum step size.

    method:     function
        Solving method. This can be a method defined above
        or any method that returns a position and time value,
        within a numpy array.

    Returns
    -------
    sol:	numpy.array(float)
        array containing the solution, with position and time values.
    """
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
    """
    Function that plots the solitions of an ODE against time.

    Parameters
    ----------
    tspan:  numpy.array(float)
        Array of the time series to calculate the solution and plot with.

    solution:   numpy.array(float)
        Array containing the solution.
    """
    for i, sol in enumerate(solution):
        plt.plot(tspan, sol, label='x{}'.format(i+1))
    
    plt.ylabel('x')
    plt.xlabel('time (s)')
    plt.legend()
    plt.show()

def get_error(tspan, solution, exact_func, exact_args):
    """
    Function that calculates the error of a numerical solution estimate,
    compared with the exact function solution.

    Parameters
    ----------
    tspan:  numpy.array(float)
        Array of the time series to calculate the solution and plot with.

    solution:   numpy.array(float)
        Array containing the solution.

    exact_func:   function
        Equialent function in its integrated form, returning a numpy array.

    exact_args:   numpy.array(float)
        Arguments for the exact function.
    """
    exact_solution = np.array([exact_func(tspan, exact_args) for t in tspan])
    abs_error = np.absolute(np.subtract(solution, exact_solution))

    return np.sum(abs_error)

def get_error_list(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, method):
    """
    Function that generates a list of errors, calculated for increasing step
    sizes.

    Parameters
    ----------
    tspan:  numpy.array(float)
        Array of the time series to calculate the solution and plot with.

    func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    x0: 	numpy.array(float)
        Position value to solve from.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.   

    exact_func:   function
        Equialent function in its integrated form, returning a numpy array.

    exact_args:   numpy.array(float)
        Arguments for the exact function.

    t0:     float
    	First time value to solve from.

    tend:   float
        Time value to solve to.

    deltat_min:	float
		Minimum step size.

	deltat_max:	float
		Maximum step size.
	
    deltat_num:	float
		Number of step sizes to calculate errors for.

    method:     function
        Solving method. This can be a method defined above
        or any method that returns a position and time value,
        within a numpy array.

    Returns
    -------
    error_list    list
        List of the errors.

    delta_t_list   list
        List of the step sizes.
    """
    delta_t_list = np.logspace(deltat_min, deltat_max, deltat_num)
    error_list = []
    for delta_t in delta_t_list:
        sol = solve_ode(func, x0, args, tspan, delta_t, method)
        error_list.append(get_error(tspan, sol, exact_func, exact_args))

    return error_list, delta_t_list

def plot_errors(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, methods):
    """
    Function that generates a list of errors, calculated for increasing step
    sizes.

    Parameters
    ----------
    tspan:  numpy.array(float)
        Array of the time series to calculate the solution and plot with.

    func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    x0: 	numpy.array(float)
        Position value to solve from.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.   

    exact_func:   function
        Equialent function in its integrated form, returning a numpy array.

    exact_args:   numpy.array(float)
        Arguments for the exact function.

    deltat_min:	float
		Minimum step size.

	deltat_max:	float
		Maximum step size.
	
    deltat_num:	float
		Number of step sizes to calculate errors for.

    methods:     function
        List of methods to calculate and plot the errors for.
    """
    for m in methods:
        method = m
        error_list, deltat_list = get_error_list(tspan, func, x0, args, exact_func, exact_args, deltat_min, deltat_max, deltat_num, method)
        plt.loglog(deltat_list, error_list, '-', label=method.__name__)

    plt.ylabel('Error')
    plt.xlabel('Step size')
    plt.legend()
    plt.show()

def get_time(func, x0, args, tspan, deltat_max, method):
    """
    Function that generates series of numerical solution estimates from
    an initial position value, for a series of time values.

    Parameters
    ----------
    func: 	function
		ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    x0: 	numpy.array(float)
        Position value to solve from.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.   

    tspan:  np.array(float)
        Array of the time series to calculate the solution and plot with.

	deltat_max:	float
		Maximum step size.

    method:     function
        Solving method. This can be a method defined above
        or any method that returns a position and time value,
        within a numpy array.

    Returns
    -------
    time:	float
        Time taken for the solution to be calculated.
    """
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

    methods = [euler_step, rk4_step, rk5_step]
    method = methods[0]

    sol = solve_ode(func, x0, args, tspan, deltat_max, method)
    for m in methods:
        time = get_time(func, x0, args, tspan, deltat_max, m)
        #print(time)
        
    #plot_solution(tspan, sol)
    #plot_errors(tspan, func, x0, args, exact_func, exact_args, -6, -1, 20, methods)

if __name__ == "__main__":

    main()
