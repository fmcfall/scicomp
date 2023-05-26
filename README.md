# scicomp
emat30008_SC

## Scientific Computing Coursework
Finn McFall

## Summary
General numerical continuation code that can track, under variations of a parameter in the system:

- limit cycle oscillations of arbitrary ordinary differential equations (of any number of dimensions)
- steady-states of second-order diffusive PDEs.

## Brief Overview of Files
1. all_ode.py, all_pde.py
Contain some ODE and PDE functions to be used within the software

2. solve_ode.py
Functions necessary to solve ODEs for plotting and error analysis. Includes plotting functions, time analysis function, and 3 solving methods:
- Euler's method
- 4th order Runga-Kutta method
- 5th order Runga-Kutta method

3. solve_pde.py
Functions necessary to second order diffusive solve PDEs. Includes plotting functions, time analysis function, and 3 solving methods:
- Forward Euler method
- Backward Euler method
- Crank Nicholson method
The user can also choose to use Dirichlet boundary conditions, under the `dirichlet_conditions` argument.

4. shooting.py
Functions for isolating limit cycle osciallations of arbitrary ODEs and solving using numerical shooting. Includes separate functions for isolating a single limit cycle using `argmax`, numerical shooting to obtain a strong numerical estimate for the solution and period, and plotting.

5. continuation.py
Functions for parameter continuation, including methods of natural and pseudo-arclength continuation. Includes plotting function.

6. test.py
Testing functions.

7. requirements.txt
Libraries used, necessary for pytests to run.

## Thanks for using this software! Enjoy!
