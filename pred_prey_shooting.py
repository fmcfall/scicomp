from re import M
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from math import isclose

def dXdt(t, X, a, b, d):

    return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])

''' 
predator-prey shooting method 
'''

def get_ode_data(ode, u0, args):

    x0 = u0[:-1]
    t = u0[-1]
    t_span = (0, t)

    data = solve_ivp(ode, t_span, x0, max_step=1e-2, args=args)
    y_data = data.y
    t_data = data.t

    return np.array([y_data, t_data])

def get_period(y_data, t_data):
    
    # argrelectrema returns array of indices of maxima 
    maxima = argrelmax(y_data)[0]

    c = 0
    while not isclose(y_data[maxima[c]], y_data[maxima[c+1]], rel_tol=1e-4):
        c += 1

    print(y_data[maxima[c]],y_data[maxima[c+1]])

    period = t_data[maxima[c+1]] - t_data[maxima[c]]
    
    return period

def main():

    args = np.array([1, 0.2, 0.1])
    u0 = [1, 0.5, 200]
    p = get_ode_data(dXdt, u0, args)

    y = p[0]
    prey = y[0]
    predator = y[1]
    t = p[1]

    print(get_period(predator, t))

    f1 = plt.figure()
    plt.plot(t, prey, 'r-', label='Prey')
    plt.plot(t, predator  , 'b-', label='Predator')
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.show()

if __name__ == '__main__':
    
    main()

