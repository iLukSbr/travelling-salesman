from abc import ABC, abstractmethod
import numpy as np
import random
from numba import jit
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class Algorithm(ABC):
    def __init__(self, cities, use_gpu=False):
        """
        Inicializa a classe base com as cidades e calcula a matriz de distâncias.

        Args:
            cities (list): Lista de coordenadas das cidades [(x1, y1), (x2, y2), ...].
            use_gpu (bool): Define se a GPU será usada para cálculos (se disponível).
        """
        if not cities:
            raise ValueError("A lista de cidades não pode estar vazia.")
        self.cities = self.normalize_coordinates(np.array(cities))
        self.n = len(cities)
        self.use_gpu = use_gpu
        self.dist_matrix = self.compute_distance_matrix()

    @staticmethod
    def normalize_coordinates(cities):
        """
        Normaliza as coordenadas para garantir que todas sejam positivas.

        Args:
            cities (np.ndarray): Array de coordenadas [(x1, y1), (x2, y2), ...].

        Returns:
            np.ndarray: Array de coordenadas normalizadas.
        """
        min_x = np.min(cities[:, 0])
        min_y = np.min(cities[:, 1])
        normalized_cities = cities - [min_x, min_y]

        # Log para verificar os valores normalizados
        # print("Coordenadas normalizadas:\n", normalized_cities)

        return normalized_cities

    def compute_distance_matrix(self):
        """
        Precomputa a matriz de distâncias para todos os pares de cidades.

        Returns:
            np.ndarray: Matriz de distâncias.
        """
        if self.use_gpu and GPU_AVAILABLE:
            dist_matrix = self._compute_distance_matrix_gpu()
        else:
            dist_matrix = self._compute_distance_matrix_cpu()

        # Garantir que a matriz seja simétrica
        if not np.allclose(dist_matrix, dist_matrix.T):
            raise ValueError("A matriz de distâncias não é simétrica.")

        # Log para verificar os valores da matriz
        # print("Matriz de distâncias:\n", dist_matrix)

        return dist_matrix

    def _compute_distance_matrix_cpu(self):
        """Calcula a matriz de distâncias usando a CPU."""
        x = self.cities[:, 0]
        y = self.cities[:, 1]
        return np.sqrt((x[:, None] - x) ** 2 + (y[:, None] - y) ** 2)

    def _compute_distance_matrix_gpu(self):
        """Calcula a matriz de distâncias usando a GPU."""
        cities_gpu = cp.array(self.cities)
        x = cp.expand_dims(cities_gpu[:, 0], axis=1)
        y = cp.expand_dims(cities_gpu[:, 1], axis=1)
        dist_matrix = cp.sqrt((x - x.T) ** 2 + (y - y.T) ** 2)
        return cp.asnumpy(dist_matrix)

    @staticmethod
    @jit(nopython=True)
    def calculate_total_distance(route, dist_matrix):
        """
        Calcula a distância total de uma rota usando a matriz de distâncias.

        Args:
            route (list): Sequência de índices representando a rota.
            dist_matrix (np.ndarray): Matriz de distâncias precomputada.

        Returns:
            float: Distância total da rota.
        """
        total = 0.0
        n = len(route)
        for i in range(n - 1):
            total += dist_matrix[route[i], route[i + 1]]
        total += dist_matrix[route[n - 1], route[0]]

        # Validação para custos negativos
        if total < 0:
            raise ValueError(f"Custo total negativo detectado: {total}")

        return total

    @staticmethod
    def nearest_neighbor(dist_matrix):
        """
        Gera uma solução inicial usando o heurístico do vizinho mais próximo.

        Args:
            dist_matrix (np.ndarray): Matriz de distâncias precomputada.

        Returns:
            np.ndarray: Rota inicial gerada.
        """
        n = dist_matrix.shape[0]
        unvisited = set(range(n))
        current = random.choice(list(unvisited))
        route = [current]
        unvisited.remove(current)
        while unvisited:
            next_city = min(unvisited, key=lambda city: dist_matrix[current, city])
            route.append(next_city)
            unvisited.remove(next_city)
            current = next_city
        return np.array(route)

    @abstractmethod
    def solve(self):
        """
        Método abstrato para resolver o problema.
        Deve ser implementado pelas subclasses.
        """
        pass
