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
    def __init__(self, dimension, lower_bound, upper_bound, number_of_individuals, number_of_gen_cycles, fncObject):
        self.dimension = dimension
        self.lB = lower_bound  # we will use the same bounds for all parameters
        self.uB = upper_bound
        self.NP = number_of_individuals
        self.g_maxim = number_of_gen_cycles
        self.gamma = 0.5
        self.alpha = 0.3
        self.parameters = np.zeros(self.dimension)  # solution parameters
        self.f = np.inf  # objective function evaluation
        self.fncObject = fncObject

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
        print(len(best_xxs[0]), len(best_yys[0]), len(best_zzs[0]))

        self.draw(self.lB, self.uB, fnc, ax)
        for i in range(len(best_xxs[0])):
            point, = ax.plot([best_xxs[i][0]], [best_yys[i][0]], [best_zzs[i][0]], 'o')
            points.append(point)
        animate = animation.FuncAnimation(fig, self.animate, len(best_xxs), fargs=(best_xxs, best_yys, best_zzs, points), interval=30,
                                          repeat=False)
        plt.show()

    def teachNLearn(self, fnc):
        population = self.generateNeighboursUniform()
        t = 0
        solutions = []

        while self.fncObject.getEvals() < 3000:
            #teach
            populationMean = self.getPopulationMean(population)
            teacherIndex = self.getTeacherIndex(population, fnc)
            self.tryMoveMeanTowardsTeacher(population, teacherIndex, populationMean, fnc)

            #learn
            for i in range(len(population)):
                xNew = []
                r = np.random.uniform()
                j = np.random.randint(0, self.NP)
                while(j == i):
                    j = np.random.randint(0, self.NP)
                if(fnc(population[i]) < fnc(population[j])):
                    xNew = np.add(population[i], np.multiply(r, np.subtract(population[i], population[j])))
                else:
                    xNew = np.add(population[i], np.multiply(r, np.subtract(population[j], population[i])))

                self.checkBoundaries(xNew)
                if(fnc(xNew) < fnc(population[i])):
                    population[i] = np.ndarray.tolist(xNew)

            solutions.append(copy.deepcopy(population))
            t += 1
        if(self.dimension == 2):
            self.animateSolution(solutions, fnc)
        return population

    def getPopulationMean(self, population):
        mean = []
        helpArr = np.array(population)
        helpArr = helpArr.sum(axis=0)
        helpArr = np.divide(helpArr, self.NP)
        return helpArr

    def getTeacherIndex(self, population, fnc):
        bestEval = fnc(population[0])
        bestIndex = 0
        for idx, x in enumerate(population):
            tryBest = fnc(x)
            if(tryBest < bestEval):
                bestEval = tryBest
                bestIndex = idx

        return bestIndex

    def tryMoveMeanTowardsTeacher(self, population, teacherIndex, mean, fnc):
        teacherCopy = copy.deepcopy(population[teacherIndex])
        r = np.random.uniform(0,1)
        tf = np.random.randint(1,3)
        # teacherCopy = np.multiply(r, np.subtract(teacherCopy, np.multiply(tf, mean)))
        teacherCopy = np.add(teacherCopy, np.multiply(r, np.subtract(teacherCopy, np.multiply(tf, mean))))
        self.checkBoundaries(teacherCopy)
        if(fnc(teacherCopy) < fnc(population[teacherIndex])):
            population[teacherIndex] = np.ndarray.tolist(teacherCopy)

    def generateNeighboursUniform(self):
        p = []
        for xi in range(self.NP):
            pi = []
            for i in range(self.dimension):
                pi.append(np.random.uniform(self.lB,self.uB))
            p.append(pi)
        return p

    def checkBoundaries(self, learner):
        for i in range(self.dimension):
            if(learner[i] > self.uB):
                learner[i] = self.uB
            if(learner[i] < self.lB):
                learner[i] = self.lB

    def draw(self, min, max, fnc, ax):
        X = np.linspace(min, max, 200)
        Y = np.linspace(min, max, 200)
        X, Y = np.meshgrid(X, Y)
        Z = fnc([X, Y])
        ax.plot_surface(X, Y, Z, alpha=0.2)



class Function:
    def __init__(self, name):
        self.name = name
        self.numberOfEvaluations = 0

    def getEvals(self):
        return self.numberOfEvaluations

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
def getBest(population, fnc):
    bestEval = fnc(population[0])
    for x in population:
        tryBest = fnc(x)
        if(tryBest < bestEval):
            bestEval = tryBest
    return bestEval

fnc = Function("")
# testFnc = fnc
# solution = Solution(2,-500,500,5,500, fnc)
# solved = solution.teachNLearn(fnc.schwefel)
bestEvals = ""


# print(solved)
print("sphere")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.sphere
    testFnc = fnc
    solution = Solution(30, -5, 5, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("sphere.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("Schwefel...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.schwefel
    testFnc = fnc
    solution = Solution(30, -500, 500, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("schwefel.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("rosenbrock...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.rosenbrock
    testFnc = fnc
    solution = Solution(30, -5, 10, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("rosenbrock.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("rastrigin...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.rastrigin
    testFnc = fnc
    solution = Solution(30, -5.12, 5.12, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("rastrigin.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("griewank...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.griewank
    testFnc = fnc
    solution = Solution(30, -600, 600, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("griewank.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("levy...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.levy
    testFnc = fnc
    solution = Solution(30, -10, 10, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("levy.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("michalewicz...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.michalewicz
    testFnc = fnc
    solution = Solution(30, 0, np.pi, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("michalewicz.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("zakharov...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.zakharov
    testFnc = fnc
    solution = Solution(30, -5, 10, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("zakharov.csv", "w")
f.write(bestEvals)
f.close()

bestEvals = ""
print("ackley...")
for i in range(30):
    fnc = Function("")
    goFnc = fnc.ackley
    testFnc = fnc
    solution = Solution(30, -32.768, 32.768, 30, 500, fnc)
    solved = solution.teachNLearn(goFnc)
    bestEvals += str(getBest(solved, goFnc)) + "\n"

f = open("ackley.csv", "w")
f.write(bestEvals)
f.close()












