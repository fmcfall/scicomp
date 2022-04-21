import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from all_ode import *
from shooting_method import shooting

def peturb(param, delta):

    return param + delta * param

def get_secant(ode, u0, u1, args0, args1):

    sol_0 = shooting(ode, u0, args=args0)[:-1]
    sol_1 = shooting(ode, u1, args=args1)[:-1]

    return sol_1 - sol_0, u0[-1]

def pseudo_arclength(ode, u0, args, t, sol, secant):

    approx = sol + secant
    f = solve_ivp(ode, (0, t), u0, max_step=1e-2, args=args).y[:,-1]
    pseudo = np.dot(sol-approx, secant)

    return np.concatenate((f, pseudo), axis=None)

def natural_continuation(ode, u0, par0, vary_par, perturbation):

    par0[vary_par] = par0[vary_par] + perturbation

    return shooting(ode, u0, args=par0), np.array(par0)

def psuedo_continuation():
    pass

def contiuation(ode, u0, par0, vary_par=0, max_steps=100, method="psuedo", perturbation=0.1):

    n = 0
    sol = []
    pars = []
    if method == "psuedo":

        psuedo_continuation()

    if method == "natural":

        while n < max_steps:
            sol.append(u0)
            pars.append(par0)
            u0, par0 = natural_continuation(ode, u0, par0, vary_par=vary_par, perturbation=perturbation)
            n += 1

    return np.array(sol), np.array(pars)
    
u0 = [1.5, 0, 20]
par0 = [0, -1]
sol, pars = contiuation(hopf_bifurcation, u0, par0, 0, 20, method="natural")

plt.subplot(1, 2, 1)
plt.plot(pars[:,0], sol[:,0])
plt.subplot(1, 2, 2)
plt.plot(pars[:,0], sol[:,1])
plt.show()