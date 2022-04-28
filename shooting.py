import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import argrelmax
from math import isclose
from all_ode import lokta_volterra

''' 
shooting method 
'''

def get_ode_data(ode, u0, args):
    """
    Function that finds the solution to the ODE using solve_ivp.

    Parameters
    ----------
    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    u0:     numpy.array(float)
        Initial position values the ODE starts with.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.

    Returns
    -------
    y_data    np.array(float)
        Array of the position data.

    t_data    np.array(float)
        Array of the time data.
    """

    # unpack args
    x0, t = u0[:-1], u0[-1]
    t_span = (0, t)

    # solves ode and produces data
    data = solve_ivp(ode, t_span, x0, max_step=1e-2, args=args)
    y_data = data.y
    t_data = data.t

    return y_data, t_data

def limit_cycle(y_data, t_data):
    '''
    Isolates limit cycle and finds the initial conditions for one cycle.

    Parameters
    ----------
    y_data    np.array(float)
        Array of the position data.

    t_data    np.array(float)
        Array of the time data.

    Returns
    -------
    u0    numpy.array(float)
        Initial values for the isolated limit cycle.
    """
    '''
    
    # argrelectrema returns array of indices of maxima 
    maxima = argrelmax(y_data[0])[0]

    # loop finds two maxima that are the same (within 1e-4 of each other)
    c = 0

    if maxima.shape[0] > 1:
        i1, i2 = maxima[c], maxima[c+1]
        while not isclose(y_data[0,i1], y_data[0,i2], rel_tol=1e-4):
            c += 1
            if c >= maxima.shape[0] - 1:
                raise RuntimeError('Error: No limit cycle found, try increasing the period')
            i1, i2 = maxima[c], maxima[c+1]
        # calculates period between two maxima
        period = t_data[i2] - t_data[i1]
    else:
        i2 = maxima[c]
        period = t_data[-1] - t_data[0]

    # update u0 with period and initial y values
    u0 = list(y_data[:,i2])
    u0.append(period)

    return u0

def shooting(ode, u0, args):
    '''
    Function that solves an ODE using numerical shooting (fsolve).
    
    Parameters
    ----------
    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    u0:     numpy.array(float)
        Initial position values the ODE starts with. This is selected by the user,
        the function will find the limit cycle conditions automatically.

	args:	np.array(float)
		Tuple of parameter values used in the ODE.

    Returns
    -------
    solution    np.array(float)
        Array of the initial conditions for the limit cycle, with the last
        element being the period, T.
    '''    
    def G(u0, ode, args):
        """
        Function that sets up the conditions for the numerical shooting,
        including finding the phase condition.
    
        Parameters
        ----------
        u0:     numpy.array(float)
            Initial position values the ODE starts with. This is selected by the user,
            the function will find the limit cycle conditions automatically.

        ode:    function
            ODE function. The ODE function should take a time value, 
            position vector and the parameters (args). It should return
            ta numpy array.

        args:	np.array(float)
            Tuple of parameter values used in the ODE.

        Returns
        -------
        conditions    np.array(float)
            Array of the shooting conditions. The final element is the phase condition.
        """

        y0 = u0[:-1]
        t = u0[-1]

        # find F(u0, T)
        f = solve_ivp(ode, (0, t), y0, max_step=1e-2, args=args).y[:,-1]

        # find u0 - F(u0, T) = 0
        y_conditions = y0 - f

        # find phase condition, dx/dt = 0
        phase_condition = np.array(ode(t, y0, *args)[0])

        return np.concatenate((y_conditions, phase_condition), axis=None)
    
    check_inputs(ode, u0, args)

    y, t = get_ode_data(ode, u0, args)
    #print("User guess:", u0)
    u0 = limit_cycle(y, t)
    #print("Initial guess: ", u0)

    return fsolve(G, u0, args=(ode, args))

def check_inputs(ode, u0, args):
    """
    Function that tests the user inputs. If they are incorrect, an error is raised

    Parameters
    ----------
    u0:     numpy.array(float)
        Initial position values the ODE starts with. This is selected by the user,
        the function will find the limit cycle conditions automatically.

    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
        ta numpy array.

    args:	np.array(float)
        Tuple of parameter values used in the ODE.
    """
    try:
        x0 = u0[:-1]
        t = u0[-1]
        t_span = (0, t)
        solve_ivp(ode, t_span, x0, max_step=1e-2, args=args)
    except ValueError as v:
        print(f"Error {type(v)}: Wrong number of arguments or initial values for this ODE")
        quit()

def plot_solutions(ode, u0, args):
    """
    Function that plots the solutions to the shooting method. The first plot is
    the whole function and the second as an isolated limit cycle.

    Parameters
    ----------
    u0:     numpy.array(float)
        Initial position values the ODE starts with. This is selected by the user,
        the function will find the limit cycle conditions automatically.

    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
        ta numpy array.

    args:	np.array(float)
        Tuple of parameter values used in the ODE.
    """
    # get sol
    sol = shooting(ode, u0, args=args)
    # plot raw data
    plt.subplot(1, 2, 1)
    y, t = get_ode_data(ode, u0, args)
    a = y[0]
    b = y[1]
    plt.plot(t, a, 'r-', label='a')
    plt.plot(t, b  , 'b-', label='b')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('ODE solution over whole time span')
    # plot sol
    plt.subplot(1, 2, 2)
    y, t = get_ode_data(ode, sol, args)
    a = y[0]
    b = y[1]
    plt.plot(t, a, 'r-', label='a')
    plt.plot(t, b  , 'b-', label='b')
    plt.plot([0,sol[-1]], [sol[0], sol[0]], 'ro')
    plt.plot([0,sol[-1]], [sol[1], sol[1]], 'bo')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Solution (1 limit cycle)')
    plt.show()

def main():

    ode = lokta_volterra
    u0 = np.array((2, 3, 200))
    args = np.array((1, 0.2, 0.1))
    #sol = shooting(ode, u0, args)
    plot_solutions(ode, u0, args)
    
if __name__ == "__main__":

    main()
