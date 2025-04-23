import numpy as np
import random
import math
from numba import jit
from algorithms.algorithm import Algorithm

class SimulatedAnnealing(Algorithm):
    def __init__(self, cities, use_gpu=False):
        """
        Inicializa a classe com as cidades e configurações de GPU.
        """
        super().__init__(cities, use_gpu)

    @staticmethod
    @jit(nopython=True)
    def two_opt_swap(route, i, j):
        """Realiza uma troca 2-opt invertendo o segmento entre i e j."""
        new_route = route.copy()
        new_route[i:j + 1] = new_route[i:j + 1][::-1]
        return new_route

    @staticmethod
    @jit(nopython=True)
    def two_opt_delta(route, i, j, dist_matrix):
        """Calcula a mudança incremental no custo para uma troca 2-opt."""
        n = len(route)
        a = route[(i - 1) % n]
        b = route[i]
        c = route[j]
        d = route[(j + 1) % n]
        return dist_matrix[a, c] + dist_matrix[b, d] - dist_matrix[a, b] - dist_matrix[c, d]

    @staticmethod
    @jit(nopython=True)
    def three_opt_swap(route, i, j, k, case):
        """Realiza uma troca 3-opt com base no caso especificado (0-7)."""
        new_route = route.copy()
        a, b, c = i, j, k
        if case == 0:  # Sem alteração
            return new_route
        elif case == 1:  # 2-opt em a-b
            new_route[a:b + 1] = new_route[a:b + 1][::-1]
        elif case == 2:  # 2-opt em b-c
            new_route[b:c + 1] = new_route[b:c + 1][::-1]
        elif case == 3:  # 2-opt em a-c
            new_route[a:c + 1] = new_route[a:c + 1][::-1]
        elif case == 4:  # 3-opt: inverte a-b, mantém b-c
            new_route[a:b + 1] = new_route[a:b + 1][::-1]
            new_route[b:c + 1] = new_route[b:c + 1][::-1]
        elif case == 5:  # 3-opt: inverte a-b e c-d
            new_route[a:b + 1] = new_route[a:b + 1][::-1]
            temp = new_route[c:].copy()
            new_route[c:] = new_route[b:c]
            new_route[b:b + len(temp)] = temp
        elif case == 6:  # 3-opt: inverte b-c e a-b
            new_route[b:c + 1] = new_route[b:c + 1][::-1]
            new_route[a:b + 1] = new_route[a:b + 1][::-1]
        elif case == 7:  # 3-opt puro
            temp = new_route[a:b + 1].copy()
            new_route[a:a + (c - b + 1)] = new_route[b + 1:c + 1]
            new_route[a + (c - b + 1):a + (c - b + 1) + (b - a + 1)] = temp
        return new_route

    @staticmethod
    @jit(nopython=True)
    def three_opt_delta(route, i, j, k, case, dist_matrix):
        """Calcula a mudança incremental no custo para uma troca 3-opt."""
        n = len(route)
        a, b, c = route[(i - 1) % n], route[i], route[j]
        d, e, f = route[k], route[(k + 1) % n], route[(j + 1) % n]

        if case == 0:  # Sem alteração
            return 0
        elif case == 1:  # 2-opt em a-b
            return dist_matrix[a, c] + dist_matrix[b, d] - dist_matrix[a, b] - dist_matrix[c, d]
        elif case == 2:  # 2-opt em b-c
            return dist_matrix[b, e] + dist_matrix[c, f] - dist_matrix[b, c] - dist_matrix[e, f]
        elif case == 3:  # 2-opt em a-c
            return dist_matrix[a, e] + dist_matrix[c, f] - dist_matrix[a, c] - dist_matrix[e, f]
        elif case == 4:  # 3-opt: inverte a-b, mantém b-c
            return dist_matrix[a, c] + dist_matrix[b, e] + dist_matrix[d, f] - dist_matrix[a, b] - dist_matrix[c, d] - dist_matrix[e, f]
        elif case == 5:  # 3-opt: inverte a-b e c-d
            return dist_matrix[a, c] + dist_matrix[d, e] + dist_matrix[b, f] - dist_matrix[a, b] - dist_matrix[c, d] - dist_matrix[e, f]
        elif case == 6:  # 3-opt: inverte b-c e a-b
            return dist_matrix[a, d] + dist_matrix[b, e] + dist_matrix[c, f] - dist_matrix[a, b] - dist_matrix[c, d] - dist_matrix[e, f]
        elif case == 7:  # 3-opt puro
            return dist_matrix[a, e] + dist_matrix[b, f] + dist_matrix[c, d] - dist_matrix[a, b] - dist_matrix[c, d] - dist_matrix[e, f]

    def solve(self, initial_temp=1000, cooling_rate=0.95, max_iterations=10000, min_temp=1e-3):
        """
        Resolve o problema do caixeiro viajante usando Simulated Annealing.
        """
        current_solution = self.nearest_neighbor(self.dist_matrix)
        current_cost = self.calculate_total_distance(current_solution, self.dist_matrix)
        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = initial_temp
        for _ in range(max_iterations):
            if temperature < min_temp:
                break

            # Escolha aleatória entre 2-opt e 3-opt
            if random.random() < 0.5:
                i, j = sorted(random.sample(range(self.n), 2))
                delta = self.two_opt_delta(current_solution, i, j, self.dist_matrix)
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = self.two_opt_swap(current_solution, i, j)
                    current_cost += delta
            else:
                i, j, k = sorted(random.sample(range(self.n), 3))
                case = random.randint(0, 7)
                delta = self.three_opt_delta(current_solution, i, j, k, case, self.dist_matrix)
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = self.three_opt_swap(current_solution, i, j, k, case)
                    current_cost += delta

            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            temperature *= cooling_rate

        return self.format_results(best_solution, best_cost)

    def format_results(self, route, cost):
        """
        Formata os resultados para o formato unificado.
        """
        return {
            "route": route,
            "cost": cost
        }
