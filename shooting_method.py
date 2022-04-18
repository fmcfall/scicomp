import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import argrelmax
from math import isclose

def dXdt(t, X, a, b, d):

    return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])

''' 
shooting method 
'''

def get_ode_data(ode, u0, args):
    '''
    gathers data for ode using solve_ivp
    '''

    # unpack args
    x0 = u0[:-1]
    t = u0[-1]
    t_span = (0, t)

    # solves ode and produces data
    data = solve_ivp(ode, t_span, x0, max_step=1e-2, args=args)
    y_data = data.y
    t_data = data.t

    return y_data, t_data

def update_u0(y_data, t_data):
    '''
    isolates a limit cycle and returns array for u0 including y values and period
    '''
    
    # argrelectrema returns array of indices of maxima 
    maxima = argrelmax(y_data[0])[0]

    # loop finds two maxima that are the same (within 1e-4 of each other)
    c = 0
    i1, i2 = maxima[c], maxima[c+1]
    while not isclose(y_data[0,i1], y_data[0,i2], rel_tol=1):
        c += 1
        i1, i2 = maxima[c], maxima[c+1]

    # calculates period between two maxima
    period = t_data[i2] - t_data[i1]

    # update u0 with period and initial y values
    u0 = list(y_data[:,i2])
    u0.append(period)

    return u0

def shooting_method(ode, u0, args):
    '''
    general shooting method function using fsolve to find the value of G close to zero
    '''

    y, t = get_ode_data(ode, u0, args)
    print("User guess:", u0)
    u0 = update_u0(y, t)
    print("Initial guess: ", u0)

    def G(u0, ode, args):
        '''
        the vector function G
        '''

        y0 = u0[:-1]
        t = u0[-1]

        # find F(u0, T)
        f = solve_ivp(ode, (0, t), y0, max_step=1e-2, args=args).y[:,-1]

        # find u0 - F(u0, T)
        y_conditions = y0 - f

        # find phase condition, dx/dt = 0
        phase_condition = np.array(ode(t, y0, *args)[0])

        return np.concatenate((y_conditions, phase_condition), axis=None)

        plt.figure()
    
    return fsolve(G, u0, args=(ode, args))

def main():

    args = np.array([1, 0.2, 0.1])
    u0 = [1, 0.5, 200]
    sol = shooting_method(dXdt, u0, args=args)
    print("Solution: ", sol)
    print("Period: ", sol[-1])

    y, t = get_ode_data(dXdt, sol, args)
    prey = y[0]
    predator = y[1]
    plt.plot(t, prey, 'r-', label='Prey')
    plt.plot(t, predator  , 'b-', label='Predator')
    plt.plot([0,sol[-1]], [sol[0], sol[0]], 'ro') # prey boundary condition
    plt.plot([0,sol[-1]], [sol[1], sol[1]], 'bo') # predator boundary condition
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.show()

if __name__ == '__main__':
    
    main()
