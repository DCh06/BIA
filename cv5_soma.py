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
    def __init__(self, dimension, lower_bound, upper_bound):
        self.dimension = dimension
        self.lB = lower_bound
        self.uB = upper_bound
        self.PRT = 0.4
        self.pop_size = 20
        self.path_length = 3.0
        self.M_max = 100
        self.step = 0.11
        self.f = np.inf

    def animate(self, i, best_xxs, best_yys, best_zzs, points):
        for j in range(len(best_xxs[0])):
            x = best_xxs[i][j]
            y = best_yys[i][j]
            z = best_zzs[i][j]

            points[j].set_data(np.array([x, y]))
            points[j].set_3d_properties(z, 'z')
        return points

    def animateSolution(self, best_solutions, fnc):
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        best_xxs = []
        best_yys = []
        best_zzs = []
        points = []

        for best_solution in best_solutions:
            best_xs = []
            best_ys = []
            best_zs = []
            for i in range(len(best_solution)):
                best_xs.append(best_solution[i][0])
                best_ys.append(best_solution[i][1])
                best_zs.append(fnc([best_solution[i][0], best_solution[i][1]]))
            best_xxs.append(best_xs)
            best_yys.append(best_ys)
            best_zzs.append(best_zs)

        self.draw(self.lB, self.uB, fnc, ax)
        for i in range(len(best_xxs[0])):
            point, = ax.plot([best_xxs[i][0]], [best_yys[i][0]], [best_zzs[i][0]], 'o')
            points.append(point)
        animate = animation.FuncAnimation(fig, self.animate, len(best_xxs), fargs=(best_xxs, best_yys, best_zzs, points), interval=300,
                                          repeat=False)
        plt.show()

    def self_organazing_migrating_algorithm(self, fnc):
        population = self.generatePopulationUniform()
        leader = self.getLeader(population, fnc)
        populationSolution = []
        m = 0
        while m < self.M_max:
            leader = self.getLeader(population, fnc)
            for i in range(len(population)):
                population[i] = self.recalculateIndividual(population[i], leader, fnc)

            populationSolution.append(copy.deepcopy(population))
            m += 1
        if(self.is2D()):
            self.animateSolution(populationSolution, fnc)
        return leader


    def recalculateIndividual(self, individual, leader, fnc):
        t = 0
        oldIndividual = copy.deepcopy(individual)
        newIndividual = copy.deepcopy(individual)
        partialIndividual = copy.deepcopy(individual)
        while t < self.path_length:
            for j in range(self.dimension):
                rnd = np.random.uniform(0, 1)
                prtVector = 0
                if rnd < self.PRT:
                    prtVector = 1

                newIndividual[j] = np.add(oldIndividual[j], np.multiply(np.subtract(leader[j], oldIndividual[j]), (t*prtVector)))

            newIndividual = self.fixBoundaries(newIndividual)

            if(fnc(newIndividual) < fnc(partialIndividual)):
                partialIndividual = copy.deepcopy(newIndividual)
            t += self.step

        if (fnc(partialIndividual) < fnc(oldIndividual)):
            return partialIndividual

        return oldIndividual

    def generatePopulationUniform(self):
        p = []
        for xi in range(self.pop_size):
            pi = []
            for i in range(self.dimension):
                pi.append(np.random.uniform(self.lB, self.uB))
            p.append(pi)
        return p

    def is2D(self):
        if(self.dimension == 2):
            return True
        else:
            return False

    def getLeader(self, swarm, function):
        personalBest = function(swarm[0])
        personalBestIndex = 0
        for i,particle in enumerate(swarm):
            particleValue = function(particle)
            if(personalBest > particleValue):
                personalBestIndex = i
                personalBest = particleValue

        return swarm[personalBestIndex]

    def fixBoundaries(self, individual):
        for j in range(self.dimension):
            if(individual[j] < self.lB):
                individual[j] = self.lB
            elif(individual[j] > self.uB):
                individual[j] = self.uB
        return individual

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


solution = Solution(2,-5,5)
fnc = Function("")
print(solution.self_organazing_migrating_algorithm(fnc.sphere))







