import numpy as np

def simple(t, X):

    return np.array((X[0]))

def simple_exact(t, X):

    return np.array((X*np.exp(t)))

def lokta_volterra(t, X, *args):

    a, b, d = args

    return np.array([X[0] * (1 - X[0]) - (a * X[0] * X[1]) / (d + X[0]),
                    (b * X[1]) * (1 - (X[1] / X[0]))])

def hopf_bifurcation(t, X, *args):

    beta, sigma = args

    return np.array([(beta * X[0]) - X[1] + (sigma * X[0]) * (X[0]**2 + X[1]**2),
                    X[0] + (beta * X[1]) + (sigma * X[1]) * (X[0]**2 + X[1]**2)])