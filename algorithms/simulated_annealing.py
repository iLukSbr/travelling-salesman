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
    def __init__(self, cities, use_gpu=True, initial_temperature=None, cooling_rate=0.9995, max_iterations=3000, min_temp=1e-3, max_stalled_epochs=1000):
        super().__init__(cities, use_gpu)
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.min_temp = min_temp
        self.max_stalled_epochs = max_stalled_epochs

    @staticmethod
    @jit(nopython=True)
    def two_opt_swap(route, i, j):
        new_route = route.copy()
        new_route[i:j + 1] = new_route[i:j + 1][::-1]
        return new_route

    @staticmethod
    @jit(nopython=True, cache=True)
    def two_opt_delta(route, i, j, dist_matrix):
        n = len(route)
        a = route[(i - 1) % n]
        b = route[i]
        c = route[j]
        d = route[(j + 1) % n]

        delta = dist_matrix[a, c] + dist_matrix[b, d] - dist_matrix[a, b] - dist_matrix[c, d]
        return delta

    def estimate_initial_temperature(self, route):
        deltas = []
        for _ in range(100):
            i, j = sorted(random.sample(range(len(route)), 2))
            delta = self.two_opt_delta(np.array(route, dtype=np.int32), i, j, self.dist_matrix)
            if delta != 0:
                deltas.append(abs(delta))
        if deltas:
            avg_delta = np.mean(deltas)
            # print(f"Variação média de delta: {avg_delta}")
            return -avg_delta / np.log(0.8)
        return 1000  # fallback

    def solve(self):
        tracemalloc.start()
        start_time = time.time()

        # logging.info(f"Tamanho da matriz de distâncias: {len(self.dist_matrix)}x{len(self.dist_matrix[0])}")

        current_solution = list(self.nearest_neighbor(self.dist_matrix))
        current_cost = self.calculate_total_distance(current_solution, self.dist_matrix)
        best_solution = current_solution.copy()
        best_cost = current_cost

        if self.initial_temperature is None:
            self.initial_temperature = self.estimate_initial_temperature(current_solution)
            logging.info(f"Temperatura inicial estimada: {self.initial_temperature:.2f}")

        temperature = self.initial_temperature
        convergence = [best_cost]
        no_progress_count = 0

        for iteration in range(self.max_iterations):
            if temperature < self.min_temp:
                break

            i, j = sorted(random.sample(range(len(current_solution)), 2))
            if i >= j:
                continue

            delta = self.two_opt_delta(np.array(current_solution, dtype=np.int32), i, j, self.dist_matrix)

            # Registro detalhado para verificar os valores de delta, custo e índices
            # logging.debug(f"Iteração {iteration}: i={i}, j={j}, delta={delta}, current_cost={current_cost}")

            if not np.isfinite(delta):
                raise ValueError(f"Delta inválido: {delta}, i={i}, j={j}")

            new_cost = current_cost + delta

            # Adicionando log para examinar o novo custo
            # logging.debug(f"Novo custo após delta: {new_cost}")

            # Corrigir custo negativo causado por erro de ponto flutuante
            if new_cost < 0:
                if abs(new_cost) < 1e-4:
                    new_cost = 0.0  # Corrige pequenas imprecisões
                else:
                    logging.error(f"Custo total negativo inesperado: {new_cost}. Delta: {delta}, i={i}, j={j}")
                    raise ValueError(f"Custo total negativo inesperado: {new_cost}. Delta: {delta}, i={i}, j={j}")

            prob = math.exp(-delta / temperature) if delta > 0 else 1.0
            if prob > 1e10:
                raise ValueError(f"Probabilidade muito alta: {prob}")

            if delta < 0 or random.random() < prob:
                current_solution[i:j + 1] = reversed(current_solution[i:j + 1])
                current_cost = new_cost
                no_progress_count = 0
            else:
                no_progress_count += 1

            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost

            convergence.append(best_cost)

            if no_progress_count > self.max_stalled_epochs:
                logging.warning("Estagnado. Resetando rota parcialmente.")
                random.shuffle(current_solution)
                current_cost = self.calculate_total_distance(current_solution, self.dist_matrix)
                temperature = self.initial_temperature * 0.5
                no_progress_count = 0

            temperature *= self.cooling_rate

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        logging.info(f"Simulated Annealing concluído. Menor custo: {best_cost:.2f}")
        logging.info(f"Tempo de execução: {end_time - start_time:.2f} segundos")
        logging.info(f"Uso de memória: {peak / 1024 / 1024:.2f} MB")

        return {
            "route": best_solution,
            "cost": best_cost,
            "execution_time": end_time - start_time,
            "memory_usage": peak / 1024 / 1024,
            "convergence": convergence
        }

