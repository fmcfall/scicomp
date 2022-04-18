import matplotlib.pyplot as plt
import numpy as np
from all_ode import *
from shooting_method import get_ode_data, shooting

args = np.array([0.5, -1])
u0 = [1, 0.5, 100]
sol = shooting(hopf_bifurcation, u0, args=args)
print("Solution: ", sol)
print("Period: ", sol[-1])

y, t = get_ode_data(hopf_bifurcation, sol, args)
a = y[0]
b = y[1]
plt.plot(t, a, 'r-', label='a')
plt.plot(t, b  , 'b-', label='b')
plt.plot([0,sol[-1]], [sol[0], sol[0]], 'ro')
plt.plot([0,sol[-1]], [sol[1], sol[1]], 'bo')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.show()