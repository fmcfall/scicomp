import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import argrelmax
from math import isclose

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

    if maxima.shape[0] > 1:
        i1, i2 = maxima[c], maxima[c+1]
        while not isclose(y_data[0,i1], y_data[0,i2], rel_tol=0.1):
            c += 1
            if c >= maxima.shape[0] - 1:
                print("solution not found in time span")
                quit()
            print(c)
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
    general shooting method function using fsolve to find the value of G close to zero
    '''
    check_inputs(ode, u0, args)

    y, t = get_ode_data(ode, u0, args)
    #print("User guess:", u0)
    u0 = update_u0(y, t)
    #print("Initial guess: ", u0)
    
    def G(u0, ode, args):
        '''
        the vector function G
        '''

        y0 = u0[:-1]
        t = u0[-1]

        # find F(u0, T)
        f = solve_ivp(ode, (0, t), y0, max_step=1e-2, args=args).y[:,-1]

        # find u0 - F(u0, T) = 0
        y_conditions = y0 - f

        # find phase condition, dx/dt = 0
        phase_condition = np.array(ode(t, y0, *args)[0])

        return np.concatenate((y_conditions, phase_condition), axis=None)
    
    return fsolve(G, u0, args=(ode, args))

def check_inputs(ode, u0, args):

    try:
        x0 = u0[:-1]
        t = u0[-1]
        t_span = (0, t)
        solve_ivp(ode, t_span, x0, max_step=1e-2, args=args)
    except ValueError as v:
        print(f"Error {type(v)}: Wrong number of arguments or initial values for this ODE")
        quit()