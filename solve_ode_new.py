import numpy as np
import matplotlib.pyplot as plt

def main():

    def func(x,t):
        return x
    
    t = np.linspace(0,1,10)

    print(solve_to(func, 1, t[0], t[-1], 10))

def euler_step(func, x, t, h):

    return x + h * func(x, t)

def solve_to(func, x0, t0, tend, deltat_max):

    h = tend / deltat_max
    t = t0
    x = x0

    for i in range(deltat_max):
        x = euler_step(func, x, t, h)
        print(x)
        t += h

    return x

def solve_ode(func, y0, t, method):

    pass

if __name__ == "__main__":

    main()