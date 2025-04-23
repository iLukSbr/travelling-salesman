import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from .algorithm import Algorithm
import time
import tracemalloc

class Genetic(Algorithm):
    def __init__(self, cities, pop_size=200, generations=500, mutation_rate=0.02, elitism_size=4, tournament_size=5, use_gpu=True):
        """
        Inicializa a classe com as cidades e parâmetros do algoritmo genético.
        """
        super().__init__(cities, use_gpu)
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_size = min(elitism_size, pop_size // 2)
        self.tournament_size = tournament_size
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _fitness(self, route):
        """Fitness é o inverso da distância total."""
        return 1 / self.calculate_total_distance(route, self.dist_matrix)

    def _initialize_population(self):
        """Inicializa a população com uma mistura de rotas heurísticas e aleatórias."""
        population = []
        heuristic_count = self.pop_size // 5
        for _ in range(heuristic_count):
            population.append(self.nearest_neighbor(self.dist_matrix))
        for _ in range(self.pop_size - heuristic_count):
            route = list(range(self.n))
            np.random.shuffle(route)
            population.append(route)
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

        # Copia segmento do parent1 e constrói o mapeamento
        for i in range(start, end + 1):
            child[i] = parent1[i]
            mapping[parent1[i]] = parent2[i]

        # Preenche posições restantes do parent2, resolvendo conflitos
        for i in range(size):
            if child[i] == -1:
                candidate = parent2[i]
                while candidate in mapping:
                    candidate = mapping[candidate]
                child[i] = candidate
        return child

    def _swap_mutation(self, route):
        """Realiza mutação por troca com probabilidade adaptativa."""
        route = route.copy()
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                j = random.randint(0, self.n - 1)
                route[i], route[j] = route[j], route[i]
        return route

    def _two_opt_local_search(self, route, max_iterations=50):
        """Aplica busca local 2-opt para melhorar uma rota."""
        best = np.array(route)
        best_dist = self.calculate_total_distance(best, self.dist_matrix)
        for _ in range(max_iterations):
            improved = False
            for i in range(1, len(best) - 1):
                for j in range(i + 1, len(best)):
                    # Realiza a troca 2-opt
                    new_route = np.concatenate((best[:i], best[i:j][::-1], best[j:]))
                    new_dist = self.calculate_total_distance(new_route, self.dist_matrix)
                    if new_dist < best_dist:
                        best = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best.tolist()

    def _parallel_fitness(self, population):
        """Calcula o fitness da população em paralelo."""
        return list(self.executor.map(self._fitness, population))

    def solve(self):
        """Executa o algoritmo genético para encontrar a melhor rota TSP."""
        # Recria o executor para evitar problemas de shutdown
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Inicia o rastreamento de memória
        tracemalloc.start()
        start_time = time.time()

        try:
            population = self._initialize_population()
            fitnesses = self._parallel_fitness(population)
            best_route = population[np.argmax(fitnesses)]
            best_dist = self.calculate_total_distance(best_route, self.dist_matrix)

            print(
                f"[INFO] Iniciando algoritmo genético com {self.generations} gerações e população de {self.pop_size} indivíduos.")

            convergence = [best_dist]  # Lista para armazenar os custos ao longo das gerações

            for gen in range(self.generations):
                print(f"[INFO] Geração {gen + 1}/{self.generations} - Melhor custo até agora: {best_dist:.2f}")

                # Elitismo: preserva as melhores soluções
                elite_indices = np.argsort(fitnesses)[::-1][:self.elitism_size]
                new_population = [population[i].copy() for i in elite_indices]

                # Taxa de mutação adaptativa
                progress = gen / self.generations
                adaptive_mutation_rate = self.mutation_rate * (1 + progress)

                # Gera nova população
                while len(new_population) < self.pop_size:
                    parent1 = self._tournament_selection(population, fitnesses)
                    parent2 = self._tournament_selection(population, fitnesses)
                    child = self._pmx_crossover(parent1, parent2)
                    child = self._swap_mutation(child)
                    new_population.append(child)

                # Atualiza população e fitnesses
                population = new_population[:self.pop_size]
                fitnesses = self._parallel_fitness(population)

                # Aplica 2-opt ao melhor indivíduo a cada 10 gerações
                if gen % 10 == 0 or gen == self.generations - 1:
                    best_idx = np.argmax(fitnesses)
                    optimized = self._two_opt_local_search(population[best_idx])
                    opt_dist = self.calculate_total_distance(optimized, self.dist_matrix)
                    if opt_dist < self.calculate_total_distance(population[best_idx], self.dist_matrix):
                        population[best_idx] = optimized
                        fitnesses[best_idx] = self._fitness(optimized)

                # Atualiza a melhor solução
                current_best_idx = np.argmax(fitnesses)
                current_best_dist = self.calculate_total_distance(population[current_best_idx], self.dist_matrix)
                if current_best_dist < best_dist:
                    best_route = population[current_best_idx].copy()
                    best_dist = current_best_dist

                # Armazena o custo atual para o gráfico de convergência
                convergence.append(best_dist)

            # Calcula o tempo de execução e o uso de memória
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"[INFO] Algoritmo genético concluído. Melhor custo encontrado: {best_dist:.2f}")
            print(f"[INFO] Tempo de execução: {end_time - start_time:.2f} segundos")
            print(f"[INFO] Uso de memória: {peak / 1024 / 1024:.2f} MB")

            return {
                "route": best_route,
                "cost": best_dist,
                "execution_time": end_time - start_time,
                "memory_usage": peak / 1024 / 1024,
                "convergence": convergence
            }
        finally:
            # Encerra o executor no final, garantindo que ele seja liberado
            self.executor.shutdown(wait=True)

