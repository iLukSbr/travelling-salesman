import numpy as np
import random
import math
from numba import jit
from .algorithm import Algorithm
import time
import tracemalloc

class SimulatedAnnealing(Algorithm):
    def __init__(self, cities, use_gpu=False, initial_temperature=1000, cooling_rate=0.95, max_iterations=10000, min_temp=1e-3):
        """
        Inicializa a classe com as cidades, configurações de GPU e parâmetros do algoritmo.
        """
        super().__init__(cities, use_gpu)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.min_temp = min_temp

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

    def solve(self):
        """
        Resolve o problema do caixeiro viajante usando Simulated Annealing.
        """
        # Inicia o rastreamento de memória
        tracemalloc.start()
        start_time = time.time()

        current_solution = self.nearest_neighbor(self.dist_matrix)
        current_solution = list(current_solution)  # Garante que seja uma lista
        current_cost = self.calculate_total_distance(current_solution, self.dist_matrix)
        best_solution = current_solution.copy()
        best_cost = current_cost

        temperature = self.initial_temperature
        convergence = [best_cost]  # Lista para armazenar os custos ao longo das iterações

        for _ in range(self.max_iterations):
            if temperature < self.min_temp:  # Temperatura mínima para parar
                break

            # Gera uma nova solução por 2-opt
            i, j = sorted(random.sample(range(len(current_solution)), 2))
            if i >= j or j >= len(current_solution):  # Garante que i < j e índices válidos
                continue

            # Realiza a troca 2-opt
            new_solution = current_solution[:i] + current_solution[i:j + 1][::-1] + current_solution[j + 1:]
            new_cost = self.calculate_total_distance(new_solution, self.dist_matrix)

            # Aceita a nova solução com base na probabilidade de aceitação
            if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / temperature):
                current_solution = new_solution
                current_cost = new_cost

            # Atualiza a melhor solução encontrada
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            # Armazena o custo atual para o gráfico de convergência
            convergence.append(best_cost)

            # Reduz a temperatura
            temperature *= self.cooling_rate

        # Calcula o tempo de execução e o uso de memória
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"[INFO] Algoritmo Simulated Annealing concluído. Melhor custo encontrado: {best_cost:.2f}")
        print(f"[INFO] Tempo de execução: {end_time - start_time:.2f} segundos")
        print(f"[INFO] Uso de memória: {peak / 1024 / 1024:.2f} MB")

        # Retorna os resultados no formato esperado
        return {
            "route": best_solution,
            "cost": best_cost,
            "execution_time": end_time - start_time,
            "memory_usage": peak / 1024 / 1024,
            "convergence": convergence
        }

