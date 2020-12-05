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

class Individual:
    def __init__(self, functions):
        self.h = self.generateHeight()
        self.r = self.generateRadius()
        self.functions = functions
        self.Tarea = self.evaluateT()
        self.Sarea = self.evaluateS()
        self.S = []
        self.n = 0

    def evaluateT(self):
        x = self.functions.totalSurface(self.h, self.r)
        return x

    def evaluateS(self):
        return self.functions.lateralSurface(self.h, self.r)

    def generateHeight(self):
        return np.random.uniform(0, 20)

    def generateRadius(self):
        return np.random.uniform(0, 10)


class Solution:
    def __init__(self, number_of_individuals, number_of_gen_cycles, fncObject):
        self.NP = number_of_individuals
        self.g_maxim = number_of_gen_cycles
        self.fncObject = fncObject
        self.Q = []
        self.S = []
        self.n = []

    # def animate(self, i, best_xxs, best_yys, best_zzs, points):
    #     for j in range(len(best_xxs[0])):
    #         x = best_xxs[i][j]
    #         y = best_yys[i][j]
    #         z = best_zzs[i][j]
    #
    #         points[j].set_data(np.array([x, y]))
    #         points[j].set_3d_properties(z, 'z')
    #     return points
    #
    # def animateSolution(self, best_solutions, fnc):
    #     fig = plt.figure()
    #     ax = p3.Axes3D(fig)
    #
    #     best_xxs = []
    #     best_yys = []
    #     best_zzs = []
    #     points = []
    #
    #     for best_solution in best_solutions:
    #         best_xs = []
    #         best_ys = []
    #         best_zs = []
    #         for i in range(len(best_solution)):
    #             best_xs.append(best_solution[i][0])
    #             best_ys.append(best_solution[i][1])
    #             best_zs.append(fnc([best_solution[i][0], best_solution[i][1]]))
    #         best_xxs.append(best_xs)
    #         best_yys.append(best_ys)
    #         best_zzs.append(best_zs)
    #     print(len(best_xxs[0]), len(best_yys[0]), len(best_zzs[0]))
    #
    #     self.draw(self.lB, self.uB, fnc, ax)
    #     for i in range(len(best_xxs[0])):
    #         point, = ax.plot([best_xxs[i][0]], [best_yys[i][0]], [best_zzs[i][0]], 'o')
    #         points.append(point)
    #     animate = animation.FuncAnimation(fig, self.animate, len(best_xxs), fargs=(best_xxs, best_yys, best_zzs, points), interval=30,
    #                                       repeat=False)
    #     plt.show()
    # def checkBoundaries(self, learner):
    #     for i in range(self.dimension):
    #         if(learner[i] > self.uB):
    #             learner[i] = self.uB
    #         if(learner[i] < self.lB):
    #             learner[i] = self.lB
    #
    # def draw(self, min, max, fnc, ax):
    #     X = np.linspace(min, max, 200)
    #     Y = np.linspace(min, max, 200)
    #     X, Y = np.meshgrid(X, Y)
    #     Z = fnc([X, Y])
    #     ax.plot_surface(X, Y, Z, alpha=0.2)


    def teachNLearn(self, fnc):
        population = []
        population = self.generatePopulation()

        for i in range(self.NP):
            for j in range(self.NP):
                if(i == j):
                    continue
                if((population[i].Sarea >= population[j].Sarea and population[i].Tarea > population[j].Tarea)
                        or (population[i].Sarea > population[j].Sarea and population[i].Tarea >= population[j].Tarea)):
                    population[i].n += 1
                elif (population[i].Sarea < population[j].Sarea and population[i].Tarea < population[j].Tarea):
                    population[i].S.append(j)

        self.getSolutonSN(population)
        self.getSolutionQ()
        print()

    def geneticsAlgorithm(self):
        # for i in range(self.NP):
        #     if np.random.uniform() < 0.5:
        #         return (solution1.params + solution2.params) / 2
        #     else:
        #         return (solution1.params – solution2.params) / 2
        #
        #     if np.random.uniform() < 0.5:
        #         return cross + np.random.uniform(0, 1, dimension)  # cross – list of parameters
        #     else:
        #         return cross

    def generatePopulation(self):
        populaltion = []
        for i in range(self.NP):
            populaltion.append(Individual(self.fncObject))
        return populaltion

    def getSolutonSN(self, population):
        for x in population:
            self.S.append(x.S)
            self.n.append(x.n)

    def getSolutionQ(self):
        counter = sum(self.n)
        checked = []
        while(counter > 0):
            partQ = []
            for i in range(self.NP):
                if(self.n[i] == 0 and i not in checked):
                    partQ.append(i)
                    checked.append(i)
            self.Q.append(partQ)
            for x in partQ:
                for i in self.S[x]:
                    self.n[i] -= 1
            counter = sum(self.n)




class Function:
    def __init__(self, name):
        self.name = name
        self.numberOfEvaluations = 0

    def getEvals(self):
        return self.numberOfEvaluations

    def lateralSurface(self, h, r):
        s = np.sqrt(r**2 + h**2)
        S = np.pi*r*s
        return S

    def totalSurface(self, h, r):
        s = np.sqrt(r ** 2 + h ** 2)
        T = np.pi*r*(r+s)
        return T

    def sphere(self, params):
        self.numberOfEvaluations += 1
        z = 0
        for xi in params:
            z += xi ** 2
        return z

    def schwefel(self, params):
        self.numberOfEvaluations += 1
        z = 0
        dim = len(params)

        for xi in params:
            z += xi * np.sin(np.sqrt(np.abs(xi)))
        return 418.9829 * dim - z

    def rosenbrock(self, params):
        self.numberOfEvaluations += 1
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
        self.numberOfEvaluations += 1
        z = 0
        dim = len(params)
        for xi in params:
            z += xi ** 2 - 10 * np.cos(2 * np.pi * xi)
        return 10 * dim + z

    def griewank(self, params):
        self.numberOfEvaluations += 1
        counter = 1
        z = 0
        z2 = 1
        for xi in params:
            z += xi ** 2 / 4000
            z2 *= np.cos(xi / np.sqrt(counter))
            counter += 1
        return z - z2 + 1

    def levy(self, params):
        self.numberOfEvaluations += 1
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
        self.numberOfEvaluations += 1
        m = 10
        z = 0
        counter = 1
        for xi in params:
            z += np.sin(xi) * np.sin((counter * xi ** 2) / np.pi) ** (2 * 10)
            counter += 1
        return -z

    def zakharov(self, params):
        self.numberOfEvaluations += 1
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
        self.numberOfEvaluations += 1
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


fnc = Function("")
testFnc = fnc
solution = Solution(10, 500, fnc)
solved = solution.teachNLearn(fnc.schwefel)
bestEvals = ""
