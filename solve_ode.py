import numpy as np
import matplotlib.pyplot as plt

def main():

    def func(X,t):
        
        x, v = X
        dxdt = v
        dvdt = -x

        return np.array([dxdt, dvdt])
    
    t = np.linspace(0,1,10)
    X = [1, 0]

    print(solve_ode(func, X, t, method='rk4'))

def euler_step(func, x, t, h):

    return x + h * func(x, t)

def rk4_step(func, x, t, h):

    k1 = h * func(x, t)
    k2 = h * func(x+k1/2, t+h/2)
    k3 = h * func(x+k2/2, t+h/2)
    k4 = h * func(x+k3, t+h/2)
    k = (k1+2*k2+2*k3+k4)/6

    return x + k

def solve_to(func, x0, t0, tend, deltat_max):

    x = x0 # initial x
    t = t0
    h = tend / deltat_max

    for i in range(deltat_max):
        x = euler_step(func, x, t, h)
        t += h

    return x

def solve_ode(func, x0, t, method="euler"):

    solution = np.zeros((len(t),len(x0)))
    solution[0,:] = x0

    if method == "euler":
        
        for i in range(len(t)-1):

            solution[i+1] = euler_step(func, solution[i], t[i], t[i+1]-t[i])

    elif method == "rk4":

        for i in range(len(t)-1):

            solution[i+1] = rk4_step(func, solution[i], t[i], t[i+1]-t[i])

    return solution

if __name__ == "__main__":

    main()
