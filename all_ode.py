import numpy as np

class LoktaVolterra:
    def func(t, X, *args):
        a, b, d = args
        return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])
    def params():
        u0 = np.array((0.5, 1, 100))
        args = np.array((1, 0.2, 0.1))
        return u0, args

class Simple:
    def func(t, X):
        return np.array((X[0]))
    def params():
        u0 = np.array([1]) 			
        args = np.array(())
        return u0, args
    def exact(t, X):
        return np.array((X[0]*np.exp(t)))
    def exact_params():
        u0 = np.array((1))
        args = np.array(())
        return u0, args

class HopfBifurcation:
    def func(t, X, *args):
        beta, sigma = args
        return np.array([(beta * X[0]) - X[1] + (sigma * X[0]) * (X[0]**2 + X[1]**2),
                        X[0] + (beta * X[1]) + (sigma * X[1]) * (X[0]**2 + X[1]**2)])
    def params():
        u0 = np.array((0.5, 0, 20))
        args = np.array((0.5, -1))
        return u0, args

class Cubic:
    def func(t, X, c):
        return np.array((X**3 - X + c))
    def params():
        u0 = [1]
        args = [-2]
        return u0, args