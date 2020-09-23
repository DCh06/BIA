#This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random

#FNC
def sphere(*params):
    z = 0
    for xi in params:
        z +=  xi**2
    return z

def schwefel(*params):
    z = 0
    dim = len(params)

    for xi in params:
        z += xi*np.sin(np.sqrt(np.abs(xi)))
    return 418.9829*dim - z

def rosenbrock(*params):
    z = 0
    counter = 0
    dim = len(params)
    for xi in params:
        counter += 1
        z += 100*(params[counter]-xi**2)**2 + (xi-1)**2
        if(counter == dim -1):
            break
    return z

def rastrigin(*params):
    z = 0
    dim = len(params)
    for xi in params:
        z += xi**2 - 10*np.cos(2*np.pi*xi)
    return 10*dim + z

def griewank(*params):
    counter = 1
    z = 0
    z2 = 1
    for xi in params:
        z += xi**2/4000
        z2 *= np.cos(xi/np.sqrt(counter))
        counter += 1
    return z - z2 + 1

def levy(*params):
    def w(x):
        return 1 + (x - 1) / 4

    dim = len(params)
    z = 0
    counter = 0
    for xi in params:
        z += ( w(xi) - 1)**2 * (1 + 10*np.sin(np.pi*w(xi)+1)**2) + (w(params[dim-1])-1)**2 * (1 + np.sin(2*np.pi * w(params[dim-1]))**2)
        counter += 1
        if(counter >= dim - 1):
            break
    return np.sin(np.pi * w(params[0])) ** 2 + z

def michalewicz(*params):
    m = 10
    z = 0
    counter = 1
    for xi in params:
        z += np.sin(xi)*np.sin((counter*xi**2)/np.pi)**(2*10)
        counter += 1
    return -z

def zakharov(*params):
    z = 0
    z2 = 0
    z3 = 0
    counter = 1

    for xi in params:
        z += xi**2
        z2 += (0.5*counter*xi)
        z3 += (0.5*counter*xi)
        counter += 1
    return z + z2**2 + z3**4

def ackley(*params):
    a = 20
    b = 0.3
    c = np.pi/3
    z = 0
    z2 = 0
    dim = len(params)

    for xi in params:
        z += xi**2
        z2 += np.cos(c*xi)
    return -a*np.exp(-b*np.sqrt((1/dim)*z)) -np.exp((1/dim)*z2) + a + np.exp(1)
#end FNC

#helper FNC
def randrange(n, vmin, vmax):
    return (vmax - vmin)*np.random.rand(n) + vmin
#end helper FNC

#search algorithm FNC
def blindSearchMin2D(pointCount, min, max, fnc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    vhodnost0 = 1000000
    argB = 0
    X = randrange(pointCount, min, max)
    Y = randrange(pointCount, min, max)
    xl = X.tolist()
    yl = Y.tolist()
    X, Y = np.meshgrid(X, Y)
    Z = fnc(X,Y)

    for i in range(pointCount):
        for j in range (pointCount):
            arg = [xl[i],yl[j]]
            vhodnost = fnc(arg[0], arg[1])
            if(vhodnost < vhodnost0):
                vhodnost0 = vhodnost
                argB = arg

    ax.scatter(X, Y, Z, marker='o', alpha=0.2)
    print(vhodnost0)
    ax.scatter(argB[0], argB[1], vhodnost0, marker='^', alpha=1)

#end search algorithm FNC
def draw(min, max, fnc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(min, max, 200)
    print(X)
    Y = np.linspace(min, max, 200)
    X,Y = np.meshgrid(X,Y)
    print(X)
    Z = fnc(X,Y)
    print(np.amin(Z))
    ax.plot_surface(X, Y, Z)

#MAIN
# draw(-10,10,sphere)
# draw(-10,10,rosenbrock)
# draw(-500,500,schwefel)
# draw(-5.12, 5.12,rastrigin)
# draw(-6,6, griewank)
# draw(-10,10,levy)
# draw(0,np.pi,michalewicz)
# draw(-10,10, zakharov)
# draw(-40.768, 40.768, ackley)
blindSearchMin2D(10, -1000, 1000, sphere)

plt.show()


