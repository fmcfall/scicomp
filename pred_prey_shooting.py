import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def dXdt(t, X):

    a = 1
    b = 0.15
    d = 0.1

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

def main():

    args = np.array([])
    u0 = [0.5, 0.5, 100]
    p = get_ode_data(dXdt, u0, args)
    print(p)

    y = p[0]
    prey = y[0]
    predator = y[1]
    t = p[1]

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

