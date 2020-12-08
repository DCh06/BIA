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
        return np.round(x)

    def evaluateS(self):
        return np.round(self.functions.lateralSurface(self.h, self.r))

    def generateHeight(self):
        return np.random.uniform(0.001, 20)

    def generateRadius(self):
        return np.random.uniform(0.001, 10)

    def reevaluate(self):
        self.Tarea = self.functions.totalSurface(self.h, self.r)
        self.Sarea = self.functions.lateralSurface(self.h, self.r)

    def fixBoundaries(self):
        if(self.h > 20 or self.h < 0 ):
            self.h = np.random.uniform(0.001, 20)
        if(self.r > 10 or self.r < 0 ):
            self.r = np.random.uniform(0.001, 10)

    def resetIndividual(self):
        self.S = []
        self.n = 0

    def refactorIndividual(self, arr):
        self.r = arr[0]
        self.h = arr[1]
        self.reevaluate()



class Solution:
    def __init__(self, number_of_individuals, number_of_gen_cycles, fncObject):
        self.NP = number_of_individuals
        self.g_maxim = number_of_gen_cycles
        self.fncObject = fncObject
        self.Q = []
        self.S = []
        self.n = []


    def nondominatedsort(self, fnc):
        population = []
        gen = 0
        population = self.generatePopulation()
        copyOfPop = []
        allSolutions = []
        paretoSetBestSolutionsOfGenerations = []
        while(gen < self.g_maxim):
            self.Q = []
            self.S = []
            self.n = []
            self.geneticsAlgorithm(population)
            for i in range(self.NP*2):
                population[i].resetIndividual()
                for j in range(self.NP*2):
                    if(i == j):
                        continue
                    if((population[i].Sarea >= population[j].Sarea and population[i].Tarea > population[j].Tarea)
                            or (population[i].Sarea > population[j].Sarea and population[i].Tarea >= population[j].Tarea)):
                        population[i].n += 1
                    elif ((population[i].Sarea <= population[j].Sarea and population[i].Tarea < population[j].Tarea) or
                        (population[i].Sarea < population[j].Sarea and population[i].Tarea <= population[j].Tarea)):
                        population[i].S.append(j)

                    # if((population[i].Sarea <= population[j].Sarea and population[i].Tarea > population[j].Tarea) or
                    #         (population[i].Sarea < population[j].Sarea and population[i].Tarea >= population[j].Tarea)):
                    #     population[i].n += 1
                    # elif ((population[i].Sarea >= population[j].Sarea and population[i].Tarea < population[j].Tarea) or
                    #     (population[i].Sarea > population[j].Sarea and population[i].Tarea <= population[j].Tarea)):
                    #     population[i].S.append(j)

            self.getSolutonSN(population)
            self.getSolutionQ()
            copyOfPop = copy.deepcopy(population)
            allSolutions.append(copyOfPop)
            population = self.getNPforNextGen(population)

            gen += 1
            print(gen)
        paretoSetBestSolutionsOfGenerations = population
        self.drawParetoSet(paretoSetBestSolutionsOfGenerations, allSolutions)

    def createParetoSolution(self, population):
        arr = []
        helpArr = []
        for idx,x in enumerate(self.Q):
            if(idx == 1 or idx == 2):
                helpArr = []
            for i in x:
                helpArr.append(population[i])
            if(idx == 0 or idx == 1):
                arr.append(helpArr)
        arr.append(helpArr)

        return copy.deepcopy(arr)

    def getNPforNextGen(self, population):
        counter = 0
        newPop = []
        for idx, x in enumerate(self.Q):
            for i in x:
                newPop.append(population[i])
                counter += 1
                if(counter >= self.NP):
                    break
            if (counter >= self.NP):
                break
        return copy.deepcopy(newPop)

    def geneticsAlgorithm(self, population):
        for i in range(self.NP):
            newIndividual = Individual(self.fncObject)
            randomIndex = self.getDifferentRandom(i)
            #cross
            if np.random.uniform() < 0.5:
                newIndividual.h =  np.divide(np.add(population[i].h, population[randomIndex].h), 2)
                newIndividual.r =  np.divide(np.add(population[i].r, population[randomIndex].r), 2)
            else:
                newIndividual.h = np.divide(np.subtract(population[i].h, population[randomIndex].h), 2)
                newIndividual.r = np.divide(np.subtract(population[i].h, population[randomIndex].r), 2)

            if np.random.uniform() < 0.5:
                newIndividual.h = np.add(newIndividual.h, np.random.uniform(0, 1))  # cross – list of parameters
                newIndividual.r = np.add(newIndividual.r, np.random.uniform(0, 1))  # cross – list of parameters

            newIndividual.fixBoundaries()
            newIndividual.reevaluate()
            population.append(newIndividual)

    def getDifferentRandom(self, index):
        newIndex = index
        while(newIndex == index):
            newIndex = np.random.randint(0, self.NP)
        return newIndex

    def generatePopulation(self):
        population = []
        # arr = [[2,8],[1,4],[7,15],[1,2],[5,6],[3,15],[8,15],[4,1],[8,2],[5,5]]
        for i in range(self.NP):
            population.append(Individual(self.fncObject))
            # population[i].refactorIndividual(arr[i])
        return population

    def getSolutonSN(self, population):
        for x in population:
            self.S.append(x.S)
            self.n.append(x.n)

    def getSolutionQ(self):
        counter = sum(self.n)
        checked = []
        while(counter > 0):
            partQ = []
            for i in range(self.NP*2):
                if(self.n[i] == 0 and i not in checked):
                    partQ.append(i)
                    checked.append(i)
            self.Q.append(partQ)
            for x in partQ:
                for i in self.S[x]:
                    self.n[i] -= 1
            counter = sum(self.n)

    def drawParetoSet(self, bestParetoSets, allSolutions):
        set1x = []
        set1y = []
        set2x = []
        set2y = []

        for i,x in enumerate(bestParetoSets):
                set1x.append(x.r)
                set1y.append(x.h)

        for i,xx in enumerate(allSolutions):
            for indiv in xx:
                set2x.append(indiv.r)
                set2y.append(indiv.h)

        plt.scatter(set2x, set2y, color='blue')
        plt.scatter(set1x, set1y, color='green', label="Last population")
        plt.xlabel("radius")
        plt.ylabel("height")
        plt.legend()
        plt.show()




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

fnc = Function("")
testFnc = fnc
solution = Solution(10, 40, fnc)
solved = solution.nondominatedsort(fnc.schwefel)
bestEvals = ""
