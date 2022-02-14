def eulerStep(x0, h, t, f):

    return x0 + h * f(x0,t), t + h

def RK4Step():

    return 

def solveTo(x, t, t2, h, f):

    for i in range(0,int((t2)/h)):
        x, t = eulerStep(x, h, t, f)

    return x

def solveODE(X, t1, t2, h, f):

    sol = []

    for i in X:
        x = solveTo(i, t1, t2, h, f)
        sol.append(x)

    return sol

def main():

    def f(x, t):
        return x

    x1 = 1; v1 = 1
    t1 = 0; t2 = 1; h = 0.1
    x = solveODE([x1, v1], t1, t2, h, f)
    
    print(x)

if __name__ == "__main__":

    main()