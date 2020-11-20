import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

class City:
    def __init__(self, city_name):
        self.city_name = city_name
        self.x = random.uniform(0, 10)
        self.y = random.uniform(0, 10)

class Solution:
    def __init__(self, G, D):
        self.G = G
        self.D = D
        self.alpha = 1
        self.beta = 2
        self.ro = 0.5
        self.cities = ['Brno', 'Paris', 'Prague', 'Olomouc', 'Moskva', 'Berlin', 'Bern', 'Essen', 'Stockholm', 'Helsinki',
                       'Oslo', 'Washington D.C.', 'Ankara',
                       'Haag', 'Hamburg', 'Vratimov', 'Paskov', 'Ostrava', 'Frydek-Mistek', 'Opava', 'Senov', 'Havirov',
                       'Plzen', 'Hradek u Plzne', 'As']
        self.city_id = 0
        self.generated_cities = self.__generate_cities()
        self.distanceMatrix = self.__get_distance_matrix()
        self.pheromonMatrix = self.__get_pheromon_matrix()
        self.visibilityMatrix = self.__get_visibility_matrix()
        self.antsPath = self.__choose_starting_city()

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
        distance = np.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)
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
    # end init

    def __choose_starting_city(self):
        ant_start_array = []
        for i in range(self.D):
            ant_start_array.append([np.random.randint(self.D)])
        return ant_start_array

    def calculate_ant_path(self, ant_id):
        copy_visibility_matrix = copy.deepcopy(self.visibilityMatrix)
        self.recalculate_copy_of_visibility_matrix(copy_visibility_matrix, ant_id)
        for city_id in range(len(self.generated_cities)-1):
            self.antsPath[ant_id].append(self.__next_visited_node(ant_id, copy_visibility_matrix))
            self.recalculate_copy_of_visibility_matrix(copy_visibility_matrix, ant_id)
        self.antsPath[ant_id].append(self.antsPath[ant_id][0])

    def recalculate_copy_of_visibility_matrix(self, visibility_matrix, ant_id):
        last_idx = len(self.antsPath[ant_id]) - 1
        current_city = self.antsPath[ant_id][last_idx]
        for i,row in enumerate(visibility_matrix):
            visibility_matrix[i][current_city] = 0

    def __next_visited_node(self, ant_id, visibility_matrix):
        city_probability = self.get_city_probability(ant_id, visibility_matrix)
        current_probability = 0
        r = np.random.uniform()
        for i, probability in enumerate(city_probability):
            current_probability += probability
            if(r < current_probability):
                return i
        return len(visibility_matrix)-1

    def get_city_probability(self, ant_id, visibility_matrix):
        city_probability = []
        last_idx = len(self.antsPath[ant_id]) - 1
        current_city = self.antsPath[ant_id][last_idx]
        sum_of_probabilities = 0
        for city_on_row in range(len(self.visibilityMatrix[current_city])):
            sum_of_probabilities += (self.pheromonMatrix[current_city][city_on_row]**self.alpha)*(visibility_matrix[current_city][city_on_row]**self.beta)

        for city_on_row in range(len(self.visibilityMatrix[current_city])):
            x = (self.pheromonMatrix[current_city][city_on_row]**self.alpha)*(visibility_matrix[current_city][city_on_row]**self.beta)
            city_probability.append(x/sum_of_probabilities)
        return city_probability

    def __update_pheromones(self):
        for i in range(len(self.pheromonMatrix)):
            for j in range(len(self.pheromonMatrix[i])):
                self.pheromonMatrix[i][j] *= self.ro
        for antPath in self.antsPath:
            distance = self.__get_distance_of_path(antPath)
            self.__update_feromones_based_on_ant_path(distance, antPath)

    def __get_distance_of_path(self, antPath):
        distance = 0
        for idx in range(len(antPath) - 1):
            i = antPath[idx]
            j = antPath[idx + 1]
            distance += self.distanceMatrix[i][j]
        return distance

    def __update_feromones_based_on_ant_path(self, distance, antPath):
        for idx in range(len(antPath) - 1):
            i = antPath[idx]
            j = antPath[idx + 1]
            self.pheromonMatrix[i][j] += 1/distance

    def __get_index_of_best_ant(self, best_eval):
        best_ant_idx = 0
        for i in range(1, len(self.antsPath)):
            if(self.__get_distance_of_path(self.antsPath[best_ant_idx]) > self.__get_distance_of_path(self.antsPath[i])):
                best_ant_idx = i
        return i

    def ACO(self):
        best_solutions = []
        best_evals = []
        best_eval = np.inf
        curr_G = 0
        while self.G > curr_G:
            for ant_id in range(self.D):
                self.calculate_ant_path(ant_id)
            self.__update_pheromones()
            best_ant_idx = self.__get_index_of_best_ant(best_eval)
            best_current_eval = self.__get_distance_of_path(self.antsPath[best_ant_idx])
            if(best_eval > best_current_eval):
                best_eval = best_current_eval
                best_solutions.append(copy.deepcopy(self.antsPath[best_ant_idx]))
                best_evals.append(best_eval)
            self.antsPath = self.__choose_starting_city()

            print("Generation:", curr_G)

            curr_G += 1

        self.animateSolution(best_solutions, best_evals)

    def animate(self, i, best_xs,best_ys, line, text, best_evals):
        x = best_xs[i]
        y = best_ys[i]
        line.set_xdata(x)
        line.set_ydata(y)
        text.set_text(best_evals[i])

    def animateSolution(self, best_solutions_idx, best_evals):
        fig, ax = plt.subplots()
        best_xs = []
        best_ys = []
        best_solutions_cities = []
        for row_idx in best_solutions_idx:
            best_solution_cities = []
            for idx in row_idx:
                best_solution_cities.append(self.generated_cities[idx])
            best_solutions_cities.append(best_solution_cities)

        for best_solution in best_solutions_cities:
            ys = [best_solution[i].y for i in range(len(best_solution))]
            xs = [best_solution[i].x for i in range(len(best_solution))]
            ys.append(best_solution[0].y)
            xs.append(best_solution[0].x)
            best_xs.append(xs)
            best_ys.append(ys)

        ax.scatter(best_xs[0], best_ys[0])

        for c in self.generated_cities:
            plt.annotate(c.city_name, (c.x, c.y), textcoords="offset points", xytext=(0, 10))

        text = plt.text(0.95, 0.01, best_evals[0],
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes,
                color='green', fontsize=8)

        line, = ax.plot(best_xs[0], best_ys[0])
        animate = FuncAnimation(fig, self.animate, len(best_xs), fargs=(best_xs,best_ys, line, text, best_evals), interval=500,
                                      repeat=False)
        plt.show()

#MAIN
solution = Solution(100,15)
solution.ACO()


