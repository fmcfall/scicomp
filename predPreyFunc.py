from re import T
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt

def dXdt(X, t):

    a = 1
    b = 0.15
    d = 0.1

    return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])

''' odeint '''
t = np.linspace(0,200,1000)
X0 = np.array([0.6, 0.3])
X, infodict = odeint(dXdt, X0, t, full_output=True)
print(infodict['message'])
prey, predator = X.T

''' solve_ivp '''
''' swap dXdt arguments ^ '''
#t_span = [0, 200]
#X0 = [0.6, 0.3]
#X = solve_ivp(dXdt, t_span, X0, method='RK45')
#prey, predator = X.y
#t = X.t

f1 = plt.figure()
plt.plot(t, prey, 'r-', label='Prey')
plt.plot(t, predator  , 'b-', label='Predator')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.show()

# T = 25
# x(0) = 0.6
# y(0) = 0.3
# f([x,y],T) = [x(0)-x(T)
#               y(0)-y(T)
#               x(0)-0.6 ] = 0
# Fixed phase condition of x(0) = 0.6