import logging
import numpy as np
import random
import math
from numba import jit
from .algorithm import Algorithm
import time
import tracemalloc

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

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
        no_progress_count = 0  # Contador de iterações sem progresso

        for _ in range(self.max_iterations):
            # logging.info(f"Temperatura: {temperature:.2f} - Melhor custo: {best_cost:.2f}")
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
                no_progress_count = 0  # Reseta o contador ao fazer progresso
            else:
                no_progress_count += 1  # Incrementa o contador se não houver progresso

            # Atualiza a melhor solução encontrada
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost

            # Armazena o custo atual para o gráfico de convergência
            convergence.append(best_cost)

            # Interrompe se mais de 100 iterações sem progresso
            if no_progress_count > 100:
                logging.info("Interrompendo devido a falta de progresso.")
                break

            # Reduz a temperatura
            temperature *= self.cooling_rate

        # Calcula o tempo de execução e o uso de memória
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logging.info(f"Algoritmo Simulated Annealing concluído. Melhor custo encontrado: {best_cost:.2f}")
        logging.info(f"Tempo de execução: {end_time - start_time:.2f} segundos")
        logging.info(f"Uso de memória: {peak / 1024 / 1024:.2f} MB")

        # Retorna os resultados no formato esperado
        return {
            "route": best_solution,
            "cost": best_cost,
            "execution_time": end_time - start_time,
            "memory_usage": peak / 1024 / 1024,
            "convergence": convergence
        }
