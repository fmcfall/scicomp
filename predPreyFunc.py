from re import T
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dXdt(X, t=0):

    a = 1
    b = 0.5
    d = 0.1

    return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])

t = np.linspace(0,30,1000)
X0 = np.array([10, 5])
X, infodict = odeint(dXdt, X0, t, full_output=True)
print(infodict['message'])

prey, predator = X.T
f1 = plt.figure()
plt.plot(t, prey, 'r-', label='Prey')
plt.plot(t, predator  , 'b-', label='Predator')
plt.grid()
plt.legend(loc='best')
plt.xlabel('time')
plt.ylabel('population')
plt.show()