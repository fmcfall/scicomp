import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from all_ode import *
from shooting_method import limit_cycle, shooting

def update_u0(ode, u0, args, limit_cycle):

    if limit_cycle:
        return shooting(ode, u0, args)
    else:
        return fsolve(lambda u: ode(0, u, *args), np.array([u0]))

def natural_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps):

    u0 = update_u0(ode, u0, par0, limit_cycle)
    par0[vary_par] += step
    sols = np.array([u0])
    pars = np.array([par0])

    for i in range(max_steps):
        par0[vary_par] += step
        u0 = update_u0(ode, u0, par0, limit_cycle)
        sols_new = np.array([u0])
        pars_new = np.array([par0])
        sols = np.concatenate((sols, sols_new))
        pars = np.concatenate((pars, pars_new))

    return sols, pars

def update_pseudo_args(u0, u1, par0, par1, vary_par):
 
    p0 = par0[vary_par]
    p1 = par1[vary_par]
    secant_sol = u1 - u0
    pred_sol = u1 + secant_sol
    secant_par = p1 - p0
    pred_par = p1 + secant_par
    
    return secant_sol, pred_sol, secant_par, pred_par

def pseudo_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps):

    def pseudo_arclength(ode, u0, limit_cycle, par0, vary_par, secant_sol, pred_sol, secant_par, pred_par):

        y0 = u0[:-2]
        t = u0[-2]
        par = u0[-1]
        par0[vary_par] = par
        sol = solve_ivp(ode, (0, t), y0, max_step=1e-2, args=par0).y[:,-1]
        pseudo = np.dot(secant_sol, u0[:-1] - pred_sol) + np.dot(secant_par, par - pred_par)
        if limit_cycle:
            y_condition = y0 - sol
            phase_condition = np.array(ode(t, y0, *par0)[0])
            return np.concatenate((y_condition, phase_condition, pseudo), axis=None)
        else:
            y_condition = np.array(ode(t, y0, par0)[0])
            return np.concatenate((y_condition, pseudo), axis=None)

    par0[vary_par] += step
    u0 = update_u0(ode, u0, par0, limit_cycle)
    par1 = par0
    par1[vary_par] += step
    u1 = update_u0(ode, u0, par1, limit_cycle)

    sols = np.array([u0])
    pars = np.array([par0])

    sols_new = np.append(u1, par1[vary_par])
    for i in range(max_steps):
        secant_sol, pred_sol, secant_par, pred_par = update_pseudo_args(u0, u1, par0, par1, vary_par)
        sols_new = fsolve(lambda u: pseudo_arclength(ode, u, limit_cycle, par1, vary_par, secant_sol, pred_sol, secant_par, pred_par), sols_new)
        sols = np.concatenate((sols, np.array([sols_new[:-1]])))
        par0 = par1
        par1[vary_par] = sols_new[-1]
        pars = np.concatenate((pars, np.array([par1])))
        u0 = u1
        u1 = sols_new[:-1]

    return sols, pars
    
u0 = [1.5, 0, 200]
par0 = [0, -1]
max_steps = 20
sol, pars = pseudo_continuation(hopf_bifurcation, u0, True, par0, 0, 0.1, max_steps)
print(sol, pars)

plt.subplot(1, 2, 1)
plt.plot(pars[:,0], sol[:,0])
plt.subplot(1, 2, 2)
plt.plot(pars[:,0], sol[:,1])
plt.show()