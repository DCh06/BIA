# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.parameters = np.zeros(self.dimension)  # solution parameters
        self.f = np.inf  # objective function evaluation

    def similatedAnnealing(self,fnc, T0,Tmin,alpha, sigma):
        def check(list1):
            for x in list1:
                if x < self.lB or x > self.uB:
                    return False
            return True

        T = T0
        argBest = []
        vhodnostList = []

        x = []
        for i in range(self.dimension):
            x.append(np.random.uniform(self.lB, self.uB))

        while T > Tmin:
            generated_x = np.random.normal(x, sigma)
            if check(generated_x):
                x1 = generated_x
            else:
                continue

            if(fnc(x1) < fnc(x)):
                x = x1
                argBest.append(x)
                vhodnostList.append(fnc(x))
            else:
                r = np.random.uniform()
                if(r < np.e**(-(fnc(x1)-fnc(x))/T)):
                    x = x1
                    argBest.append(x)
                    vhodnostList.append(fnc(x))
            T = T*alpha

        if (self.dimension == 2):
            self.searchMinVisualization(argBest, vhodnostList, fnc)
        return fnc(x)

    def hillClimb(self, point_count, fnc, numOfNeighbours, sigma):
        argBest = []
        vhodnostList = []
        dims = []
        params = []
        arg = []
        for i in range(self.dimension):
            dims.append(self.uB)

        params = self.generateNeighbours(dims,numOfNeighbours, sigma)
        vhodnost0 = self.f

        #number of iteration in search
        for i in range(point_count):
            arg = []
            arg2 = []

            #to get [xi,yi] from [x1..xn] [y1..yn]
            for x in range(numOfNeighbours):
                arg2 = []
                for param in params:
                    arg2.append(param[x])
                arg.append(arg2)

            for argxy in arg:
                vhodnost = fnc(argxy)
                if (vhodnost < vhodnost0):
                    vhodnost0 = vhodnost
                    vhodnostList.append(vhodnost)
                    argBest.append(argxy)
            params = self.generateNeighbours(argBest[len(argBest) - 1], numOfNeighbours, sigma)

        if (self.dimension == 2):
            self.searchMinVisualization(argBest, vhodnostList, fnc)
        return vhodnost0

    def generateNeighbours(self, param, num, sigma):
        p = []
        for xi in param:
            pi = []
            for i in range(num):
                pi.append(np.random.normal(xi, sigma))
            p.append(pi)
        return p

    def blindSearch(self, point_count, fnc):
        argBest = []
        vhodnostList = []
        params = []
        for x in range(self.dimension):
            params.append(self.randrange(point_count, self.lB, self.uB))

        vhodnost0 = self.f
        for i in range(point_count):
            arg = []
            for param in params:
                arg.append(param[i])
                #
            vhodnost = fnc(arg)
            if (vhodnost < vhodnost0):
                vhodnost0 = vhodnost
                vhodnostList.append(vhodnost)
                argBest.append(arg)

        if(self.dimension == 2):
            self.searchMinVisualization(argBest,vhodnostList,fnc)
        return vhodnost0

    def updatePoints(self,n,x,y,z, point):
        point.set_data(np.array([x[n], y[n]]))
        point.set_3d_properties(z[n], 'z')
        return point

    def searchMinVisualization(self,points, z, fnc):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        x = []
        y = []

        self.draw(self.lB, self.uB, fnc, ax)
        for i in range(len(points)):
            x.append(points[i][0])
            y.append(points[i][1])
        point, = ax.plot(x[0], y[0], z[0], '^')
        line_ani = animation.FuncAnimation(fig, self.updatePoints, len(x),interval=50, fargs=(x, y, z, point), repeat=False)
        plt.show()

    def draw(self, min, max, fnc, ax):
        X = np.linspace(min, max, 200)
        Y = np.linspace(min, max, 200)
        X, Y = np.meshgrid(X, Y)
        Z = fnc([X, Y])
        ax.plot_surface(X, Y, Z, alpha=0.2)

    def randrange(self, n, vmin, vmax):
        return (vmax - vmin) * np.random.rand(n) + vmin

class Function:
    def __init__(self, name):
        self.name = name

    def sphere(self, params):
        z = 0
        for xi in params:
            z += xi ** 2
        return z

    def schwefel(self, params):
        z = 0
        dim = len(params)

        for xi in params:
            z += xi * np.sin(np.sqrt(np.abs(xi)))
        return 418.9829 * dim - z

    def rosenbrock(self, params):
        z = 0
        counter = 0
        dim = len(params)
        for xi in params:
            counter += 1
            z += 100 * (params[counter] - xi ** 2) ** 2 + (xi - 1) ** 2
            if (counter == dim - 1):
                break
        return z

    def rastrigin(self, params):
        z = 0
        dim = len(params)
        for xi in params:
            z += xi ** 2 - 10 * np.cos(2 * np.pi * xi)
        return 10 * dim + z

    def griewank(self, params):
        counter = 1
        z = 0
        z2 = 1
        for xi in params:
            z += xi ** 2 / 4000
            z2 *= np.cos(xi / np.sqrt(counter))
            counter += 1
        return z - z2 + 1

    def levy(self, params):
        def w(x):
            return 1 + (x - 1) / 4

        dim = len(params)
        z = 0
        counter = 0
        for xi in params:
            z += (w(xi) - 1) ** 2 * (1 + 10 * np.sin(np.pi * w(xi) + 1) ** 2) + (w(params[dim - 1]) - 1) ** 2 * (
                        1 + np.sin(2 * np.pi * w(params[dim - 1])) ** 2)
            counter += 1
            if (counter >= dim - 1):
                break
        return np.sin(np.pi * w(params[0])) ** 2 + z

    def michalewicz(self, params):
        m = 10
        z = 0
        counter = 1
        for xi in params:
            z += np.sin(xi) * np.sin((counter * xi ** 2) / np.pi) ** (2 * 10)
            counter += 1
        return -z

    def zakharov(self, params):
        z = 0
        z2 = 0
        z3 = 0
        counter = 1

        for xi in params:
            z += xi ** 2
            z2 += (0.5 * counter * xi)
            z3 += (0.5 * counter * xi)
            counter += 1
        return z + z2 ** 2 + z3 ** 4

    def ackley(self, params):
        a = 20
        b = 0.3
        c = np.pi / 3
        z = 0
        z2 = 0
        dim = len(params)

        for xi in params:
            z += xi ** 2
            z2 += np.cos(c * xi)
        return -a * np.exp(-b * np.sqrt((1 / dim) * z)) - np.exp((1 / dim) * z2) + a + np.exp(1)

# MAIN
solution = Solution(2,-10,10)
# solution = Solution(2,-100,100)

fnc = Function("")
# solution.blindSearch(522, fnc.sphere)
# solution.hillClimb(522, fnc.sphere, 5, 0.5)
print(solution.similatedAnnealing(fnc.sphere, 100, 0.5, 0.95, 0.5))






