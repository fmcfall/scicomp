import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from all_ode import *
from shooting import shooting

def update_u0(ode, u0, args, limit_cycle):
    """
    Function to get initial conditions from some initial parameters. Works for ODE
    systems with and without limit cycles.
    
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

    limitCycle:     Bool
        When true, function will solve using shooting method.

    Returns
    -------

    u0:     numpy.array(float)
        New initial position values the ODE starts with.
    """
    if limit_cycle:
        return shooting(ode, u0, args)
    else:
        return fsolve(lambda u: ode(0, u, *args), np.array([u0]))

def natural_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps):
    """
    Function to find a solution at each parameter value using the natural continuation
    method.
    
    Parameters
    ----------
    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    u0:     numpy.array(float)
        Initial position values the ODE starts with.

    limitCycle:     Bool
        When true, function will solve using shooting method.

	par0:	np.array(float)
		Tuple of initial parameter values used in the ODE.

    vary_par:   int
        Index of the paramater to vary in the par0 argument.

    step:   float
        The step by which to pertubate the parameter by.

    max_steps:   int
        The maximum number of pertubations.

    Returns
    -------
    sols:     numpy.array(float)
        Array of solutions at each parameter value.

    pars:     numpy.array(float)
        Array of parameters.
    """
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
    """
    Function to update the pseudo arguments.
    
    Parameters
    ----------
    u0:     numpy.array(float)
        First solution values for the ODE.

    u1:     numpy.array(float)
        Second solution values for the ODE.

	par0:	np.array(float)
		First tuple of parameter values used in the ODE.

	par1:	np.array(float)
		First tuple of parameter values used in the ODE.

    Returns
    -------
    secant_sol:  np.array(float)
        Secant solution.

    pred_sol:    np.array(float)
        Predicted solution

    secant_par:   np.array(float)
        Secant parameter

    pred_par:     np.array(float)
        Predicted parameter
    """
    p0 = par0[vary_par]
    p1 = par1[vary_par]
    secant_sol = u1 - u0
    pred_sol = u1 + secant_sol
    secant_par = p1 - p0
    pred_par = p1 + secant_par
    
    return secant_sol, pred_sol, secant_par, pred_par

def pseudo_continuation(ode, u0, limit_cycle, par0, vary_par, step, max_steps):
    """
    Function to find a solution at each parameter value using the pseudo continuation
    method.
    
    Parameters
    ----------
    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    u0:     numpy.array(float)
        Initial position values the ODE starts with.

    limitCycle:     Bool
        When true, function will solve using shooting method.

	par0:	np.array(float)
		Tuple of initial parameter values used in the ODE.

    vary_par:   int
        Index of the paramater to vary in the par0 argument.

    step:   float
        The step by which to pertubate the parameter by.

    max_steps:   int
        The maximum number of pertubations.

    Returns
    -------
    sols:     numpy.array(float)
        Array of solutions at each parameter value.

    pars:     numpy.array(float)
        Array of parameters.
    """
    def pseudo_arclength(ode, u0, limit_cycle, par0, vary_par, secant_sol, pred_sol, secant_par, pred_par):
        """
        Function to produce the pseudo arclength conditions.
        
        Parameters
        ----------
        ode:    function
            ODE function. The ODE function should take a time value, 
            position vector and the parameters (args). It should return
            ta numpy array.

        u0:     numpy.array(float)
            Initial position values the ODE starts with.

        limitCycle:     Bool
            When true, function will solve using shooting method.

        par0:	np.array(float)
            Tuple of initial parameter values used in the ODE.

        vary_par:   int
            Index of the paramater to vary in the par0 argument.

        secant_sol:    np.array(float)
        Secant solution.

        pred_sol:     np.array(float)
            Predicted solution

        secant_par:    np.array(float)
            Secant parameter

        pred_par:      np.array(float)
            Predicted parameter

        Returns
        -------
        conditions:  np.array(float)
            Pseudo continuation conditions, with the pseudo condition in the last position
            and the phase condition (if limit_cycle) in second last position.
        """
        y0 = u0[:-2]
        t = u0[-2]
        par = u0[-1]
        par0[vary_par] = par
        sol = solve_ivp(ode, (0, t), y0, max_step=0.5e-2, args=par0).y[:,-1]
        pseudo = np.dot(secant_sol, u0[:-1] - pred_sol) + np.dot(secant_par, par - pred_par)
        if limit_cycle:
            y_condition = y0 - sol
            phase_condition = np.array(ode(t, u0[:-1], *par0)[0])
            return np.concatenate((y_condition, phase_condition, pseudo), axis=None)
        else:
            y_condition = np.array(ode(0, u0, *par0)[0])
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

def plot_parameter_change(ode, u0, limit_cycle, par0, vary_par, step, max_steps, method):
    """
    Function to plot the solution vs parameter value.
    
    Parameters
    ----------
    ode:    function
        ODE function. The ODE function should take a time value, 
        position vector and the parameters (args). It should return
		ta numpy array.

    u0:     numpy.array(float)
        Initial position values the ODE starts with.

    limitCycle:     Bool
        When true, function will solve using shooting method.

	par0:	np.array(float)
		Tuple of initial parameter values used in the ODE.

    vary_par:   int
        Index of the paramater to vary in the par0 argument.

    step:   float
        The step by which to pertubate the parameter by.

    max_steps:   int
        The maximum number of pertubations.

    method:    function
        Method of continuation to use: natural or pseudo.
    """
    sol, pars = method(ode, u0, limit_cycle, par0, vary_par, step, max_steps)
    plt.plot(pars[:,0], sol[:,0],'k')
    plt.xlabel('Varying Parameter, c')
    plt.ylabel('x')
    plt.show()

def main():
    ode = hopf_bifurcation
    u0 = np.array((0.5, 0, 20))
    limit_cycle = True
    par0 = np.array([0.5,-1])
    vary_par = 0
    step = 0.05
    max_steps = 20
    methods = [natural_continuation, pseudo_continuation]
    
    plot_parameter_change(ode, u0, limit_cycle, par0, vary_par, step, max_steps, methods[0])

if __name__=="__main__":
    main()
