import logging
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from .algorithm import Algorithm
import time
import tracemalloc
import os

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class Genetic(Algorithm):
    def __init__(self, cities, pop_size=100, generations=3000, mutation_rate=0.05, elitism_size=4, tournament_size=3, use_gpu=True, max_stalled_epochs=1000):
        super().__init__(cities, use_gpu)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_size = min(elitism_size, pop_size // 2)
        self.tournament_size = tournament_size
        self.num_cores = os.cpu_count()
        logging.info(f"Número de núcleos disponíveis no CPU: {self.num_cores}")
        self.executor = ThreadPoolExecutor(max_workers=self.num_cores)
        self.max_stalled_epochs = max_stalled_epochs

    def _fitness(self, route):
        """Fitness é o inverso da distância total."""
        return 1 / self.calculate_total_distance(route, self.dist_matrix)

    def _initialize_population(self):
        """Inicializa a população com maior diversidade."""
        population = []
        heuristic_count = self.pop_size // 10  # Reduzi a proporção de rotas heurísticas
        for _ in range(heuristic_count):
            population.append(self.nearest_neighbor(self.dist_matrix))
        for _ in range(self.pop_size - heuristic_count):
            route = list(range(self.n))
            np.random.shuffle(route)
            population.append(route)
        logging.info(f"População inicial gerada com {len(population)} indivíduos.")
        return population

    def _tournament_selection(self, population, fitnesses):
        """Seleciona um pai usando seleção por torneio."""
        tournament = random.sample(list(zip(population, fitnesses)), self.tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def _pmx_crossover(self, parent1, parent2):
        """Crossover parcialmente mapeado (PMX) para TSP."""
        size = self.n
        start, end = sorted(np.random.randint(0, size, 2))
        child = [-1] * size
        mapping = {}

        for i in range(start, end + 1):
            child[i] = parent1[i]
            mapping[parent1[i]] = parent2[i]

        for i in range(size):
            if child[i] == -1:
                candidate = parent2[i]
                while candidate in mapping:
                    candidate = mapping[candidate]
                child[i] = candidate
        return child

    def _parallel_fitness(self, population):
        """Calcula o fitness da população em paralelo."""
        return list(self.executor.map(self._fitness, population))

    def _swap_mutation(self, route, mutation_rate):
        """Realiza mutação por troca com taxa adaptativa."""
        route = route.copy()
        for i in range(self.n):
            if random.random() < mutation_rate:
                j = random.randint(0, self.n - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def solve(self):
        """Executa o algoritmo genético para encontrar a melhor rota TSP."""
        self.executor = ThreadPoolExecutor(max_workers=self.num_cores)
        tracemalloc.start()
        start_time = time.time()

        try:
            population = self._initialize_population()
            fitnesses = self._parallel_fitness(population)
            best_route = population[np.argmax(fitnesses)]
            best_dist = self.calculate_total_distance(best_route, self.dist_matrix)

            logging.info(f"Iniciando algoritmo genético com {self.generations} gerações e população de {self.pop_size} indivíduos.")
            convergence = [best_dist]
            no_progress_count = 0

            for gen in range(self.generations):
                elite_indices = np.argsort(fitnesses)[::-1][:self.elitism_size]
                new_population = [population[i].copy() for i in elite_indices]

                adaptive_mutation_rate = self.mutation_rate * (1 + gen / self.generations * 2)  # Aumenta mais rápido
                while len(new_population) < self.pop_size:
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    child = self._pmx_crossover(parent1, parent2)
                    child = self._swap_mutation(child, adaptive_mutation_rate)
                    new_population.append(child)

                population = new_population[:self.pop_size]
                fitnesses = self._parallel_fitness(population)

                current_best_idx = np.argmax(fitnesses)
                current_best_dist = self.calculate_total_distance(population[current_best_idx], self.dist_matrix)
                if current_best_dist < best_dist:
                    best_route = population[current_best_idx].copy()
                    best_dist = current_best_dist
                    no_progress_count = 0
                else:
                    no_progress_count += 1

                convergence.append(best_dist)

                if no_progress_count > self.max_stalled_epochs:
                    logging.info("Interrompendo devido à falta de progresso.")
                    break

            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            logging.info(f"Algoritmo genético concluído. Menor custo encontrado: {best_dist:.2f}")
            logging.info(f"Tempo de execução: {end_time - start_time:.2f} segundos")
            logging.info(f"Uso de memória: {peak / 1024 / 1024:.2f} MB")

            return {
                "route": best_route,
                "cost": best_dist,
                "execution_time": end_time - start_time,
                "memory_usage": peak / 1024 / 1024,
                "convergence": convergence
            }
        finally:
            self.executor.shutdown(wait=True)
