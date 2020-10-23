# This import registers the 3D projection, but is otherwise unused.
import copy
from time import sleep

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

class Solution:
    def __init__(self, dimension, lower_bound, upper_bound, number_of_individuals, number_of_gen_cycles):#np = 4, gmax = 10
        self.dimension = dimension
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.NP = number_of_individuals
        self.g_maxim = number_of_gen_cycles
        self.F = 0.5
        self.CR = 0.5
        self.parameters = np.zeros(self.dimension)  # solution parameters
        self.f = np.inf  # objective function evaluation

    def animate(self, i, best_xs, best_ys, line):
        x = best_xs[i]
        y = best_ys[i]
        line.set_xdata(x)
        line.set_ydata(y)

    def animateSolution(self, best_solutions, fnc):
        fig, ax = plt.subplots()
        best_xxs = []
        best_yys = []
        best_zzs = []
        for best_solution in best_solutions:
            best_xs = []
            best_ys = []
            best_zs = []
            for x,y in best_solution:
                best_xs.append(x)
                best_ys.append(y)
                best_zs.append(fnc(x,y))



            exit()

        # for best_solution in best_solutions:
        #     print(best_solution)
        #     ys = [best_solution[i][1] for i in range(len(best_solution))]
        #     xs = [best_solution[i][0] for i in range(len(best_solution))]
        #     ys.append(best_solution[0][1])
        #     xs.append(best_solution[0][0])
        #     best_xs.append(xs)
        #     best_ys.append(ys)

        # ax.scatter(best_xs[0], best_ys[0])
        #
        # for c in self.generated_cities:
        #     plt.annotate(c.city_name, (c.x, c.y), textcoords="offset points", xytext=(0, 10))
        #
        # point, = ax.plot(best_xs[0], best_ys[0])
        # animate = animation.FuncAnimation(fig, self.animate, len(best_xs), fargs=(best_xs, best_ys, point), interval=300,
        #                                   repeat=False)
        # plt.show()

    def differential_evolution(self, fnc):
        def check(list1):
            for x in list1:
                if x < self.lB or x > self.uB:
                    return False
            return True
        draw_evolution = []
        pop = self.generateNeighboursUniform()
        g = 0

        while g < self.g_maxim:
            new_pop = copy.deepcopy(pop)  # new generation
            for i, x in enumerate(pop):  # x is also denoted as a target vector
                r1,r2,r3 = 0,0,0
                while True:
                    r1 = np.random.randint(0, self.NP)
                    if(r1 != i):
                        break
                while True:
                    r2 = np.random.randint(0, self.NP)
                    if (r2 != i and r2 != r1):
                        break
                while True:
                    r3 = np.random.randint(0, self.NP)
                    if (r3 != i and r3 != r1 and r3 != r2):
                        break

                # v = np.add((map(lambda k: k * self.F, (np.subtract(pop[r1], pop[r2])))) + pop[r3])  # mutation vector. TAKE CARE FOR BOUNDARIES!
                p = (np.subtract(pop[r1], pop[r2]))
                map(lambda k: k * self.F, p)
                v = np.add(p, pop[r3])

                u = np.zeros(self.dimension)  # trial vector
                j_rnd = np.random.randint(0, self.dimension)

                for j in range(self.dimension):
                    if np.random.uniform() < self.CR or j == j_rnd:
                        u[j] = v[j]  # at least 1 parameter should be from a mutation vector v
                    else:
                        u[j] = pop[i][j]

                f_u = fnc(u)

                if f_u <= fnc(pop[i]):  # We always accept a solution with the same fitness as a target vector
                    new_pop[i] = u
            pop = new_pop
            draw_evolution.append(new_pop)
            g += 1

        # if (self.dimension == 2):
        #     self.searchMinVisualization(argBest, vhodnostList, fnc)
        # return fnc(x)
        if(self.dimension == 2):
            self.animateSolution(draw_evolution,fnc)
        return pop

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

    def generateNeighboursUniform(self):
        p = []
        for xi in range(self.NP):
            pi = []
            for i in range(self.dimension):
                pi.append(np.random.uniform(self.lB,self.uB))

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
solution = Solution(2,-10,10,20,200)
# solution = Solution(2,-100,100)

fnc = Function("")
# solution.blindSearch(522, fnc.sphere)
# solution.hillClimb(522, fnc.sphere, 5, 0.5)
print(solution.differential_evolution(fnc.sphere))







