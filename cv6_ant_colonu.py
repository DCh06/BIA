import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class City:
    def __init__(self, city_name):
        self.city_name = city_name
        self.x = random.uniform(0, 500)
        self.y = random.uniform(0, 500)

class Solution:
    def __init__(self, NP, G, D):
        self.NP = NP
        self.G = G
        self.D = D
        self.cities = ['Brno', 'Paris', 'Prague', 'Olomouc', 'Moskva', 'Berlin', 'Bern', 'Essen', 'Stockholm', 'Helsinki',
                       'Oslo', 'Washington D.C.', 'Ankara',
                       'Haag', 'Hamburg', 'Vratimov', 'Paskov', 'Ostrava', 'Frydek-Mistek', 'Opava', 'Senov', 'Havirov',
                       'Plzen', 'Hradek u Plzne', 'As']
        self.city_id = 0
        self.generated_cities = self.__generate_cities()
        self.distanceMatrix = self.__get_distance_matrix()
        self.pheromonMatrix = self.__get_pheromon_matrix()
        self.visibilityMatrix = self.__get_visibility_matrix()

        # self.new_population = []
        self.best_solution_of_generations = []

    def __generate_cities(self):
        return_cities = []
        for x in range(self.D):
            city = City(self.cities[self.city_id])
            return_cities.append(city)
            self.city_id += 1
            if(self.city_id >= len(self.cities)):
                print("Out of range")
                exit()
        return return_cities

    def __get_distance(self, city1, city2):
        distance = 0
        distance += np.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)
        return distance

    def __get_distance_matrix(self):
        distance_matrix = []
        for i,pop in enumerate(self.generated_cities):
            matrix_row = []
            for j,pop in enumerate(self.generated_cities):
                matrix_row.append(self.__get_distance(self.generated_cities[i], self.generated_cities[j]))
            distance_matrix.append(matrix_row)
        return distance_matrix

    def __get_pheromon_matrix(self):
        return [[1 for y in range(self.D)] for x in range(self.D)]

    def __get_visibility_matrix(self):
        visibility_matrix = []
        for i,row in enumerate(self.distanceMatrix):
            visibility_row = []
            for j, col in enumerate(row):
                if(col > 0):
                    visibility_row.append(1/col)
                else:
                    visibility_row.append(0)
            visibility_matrix.append(visibility_row)
        return visibility_matrix

    def __get_best_in_generation(self):
        # best_in_gen = self.population[0]
        # for i in range(1, len(self.population) - 1):
        #     if(self.__get_distance(best_in_gen) > self.__get_distance(self.population[i])):
        #         best_in_gen = self.population[i]
        # return best_in_gen
        pass

    def animate(self, i, best_xs,best_ys, line):
        x = best_xs[i]
        y = best_ys[i]
        line.set_xdata(x)
        line.set_ydata(y)

    def animateSolution(self):
        fig, ax = plt.subplots()
        best_xs = []
        best_ys = []
        for best_solution in self.best_solution_of_generations:
            ys = [best_solution[i].y for i in range(len(best_solution))]
            xs = [best_solution[i].x for i in range(len(best_solution))]
            ys.append(best_solution[0].y)
            xs.append(best_solution[0].x)
            best_xs.append(xs)
            best_ys.append(ys)

        ax.scatter(best_xs[0], best_ys[0])

        for c in self.generated_cities:
            plt.annotate(c.city_name, (c.x, c.y), textcoords="offset points", xytext=(0, 10))

        line, = ax.plot(best_xs[0], best_ys[0])
        animate = FuncAnimation(fig, self.animate, len(best_xs), fargs=(best_xs,best_ys, line), interval=300,
                                      repeat=False)
        plt.show()

    def ant_colony(self):
        pass

    # def genetic_algorithm(self):
    #     minimum = self.__get_distance(self.population[0])
    #     for i in range(self.G):
    #         self.new_population = copy.deepcopy(self.population)
    #
    #         for j in range(self.NP):
    #             random_B = random.randint(0, len(self.population[j]) - 1)
    #             while(random_B == j):
    #                 random_B = random.randint(0, len(self.population[j]) - 1)
    #
    #             parent_A = self.population[j]
    #             parent_B = self.population[random_B]
    #
    #             offspring_AB = self.__crossover(parent_A, parent_B)
    #             if np.random.uniform() < 0.5:
    #                 offspring_AB = self.__mutate(offspring_AB)
    #
    #             if(self.__get_distance(offspring_AB) < self.__get_distance(parent_A)):
    #                 self.new_population[j] = offspring_AB
    #
    #         self.population = copy.deepcopy(self.new_population)
    #         try_min = self.__get_distance(self.__get_best_in_generation())
    #         if(minimum > try_min):
    #             minimum = try_min
    #             print(i)
    #             self.best_solution_of_generations.append(self.__get_best_in_generation())



#MAIN
solution = Solution(20,1000,10)
solution.ant_colony()
# solution.animateSolution()


