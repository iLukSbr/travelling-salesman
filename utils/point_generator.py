import logging
import random
import csv
import numpy as np

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class PointGenerator:
    @staticmethod
    def write_points_to_csv(file_path, points):
        """
        Escreve uma lista de pontos em um arquivo CSV.

        Args:
            file_path (str): Caminho para salvar o arquivo CSV.
            points (list): Lista de tuplas representando os pontos (x, y).
        """
        logging.info(f"Escrevendo {len(points)} pontos no arquivo CSV: {file_path}")
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(["x", "y"])  # Cabeçalho do CSV
            writer.writerows(points)

    @staticmethod
    def generate_random_points(n, limit=200):
        """
        Gera uma lista de coordenadas aleatórias.

        Args:
            n (int): Número de pontos a serem gerados.
            limit (int): Limite para os valores das coordenadas (x, y).

        Returns:
            list: Lista de tuplas representando os pontos (x, y).
        """
        logging.info(f"Gerando {n} pontos aleatórios com limite {limit}.")
        return [(random.randint(-limit, limit), random.randint(-limit, limit)) for _ in range(n)]

    @staticmethod
    def generate_clustered_points(n, clusters=5, limit=200):
        """
        Gera uma lista de pontos agrupados em clusters.

        Args:
            n (int): Número de pontos a serem gerados.
            clusters (int): Número de clusters.
            limit (int): Limite para os valores das coordenadas (x, y).

        Returns:
            list: Lista de tuplas representando os pontos (x, y).
        """
        logging.info(f"Gerando {n} pontos agrupados em {clusters} clusters com limite {limit}.")
        points = []
        points_per_cluster = n // clusters
        for _ in range(clusters):
            cluster_center = (random.randint(-limit, limit), random.randint(-limit, limit))
            for _ in range(points_per_cluster):
                x = random.gauss(cluster_center[0], limit // 10)
                y = random.gauss(cluster_center[1], limit // 10)
                points.append((int(x), int(y)))
        return points

    @staticmethod
    def generate_uniform_points(n, limit=200):
        """
        Gera uma lista de pontos distribuídos uniformemente.

        Args:
            n (int): Número de pontos a serem gerados.
            limit (int): Limite para os valores das coordenadas (x, y).

        Returns:
            list: Lista de tuplas representando os pontos (x, y).
        """
        logging.info(f"Gerando {n} pontos distribuídos uniformemente com limite {limit}.")
        points = []
        grid_size = int(np.sqrt(n))
        step = (2 * limit) // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                x = -limit + i * step
                y = -limit + j * step
                points.append((x, y))
        return points

    @staticmethod
    def load_points_from_csv(file_path):
        """
        Carrega os pontos de um arquivo CSV e retorna uma lista de coordenadas.

        Args:
            file_path (str): Caminho do arquivo CSV.

        Returns:
            list: Lista de tuplas representando os pontos (x, y).
        """
        logging.info(f"Carregando pontos do arquivo CSV: {file_path}")
        points = []
        with open(file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            next(reader)  # Pula o cabeçalho
            for row in reader:
                x, y = map(int, row)
                points.append((x, y))
        logging.info(f"{len(points)} pontos carregados do arquivo CSV.")
        return points

    @staticmethod
    def generate_random_points_csv(file_path, n, limit=200):
        logging.info(f"Gerando pontos aleatórios e salvando no arquivo CSV: {file_path}")
        points = PointGenerator.generate_random_points(n, limit)
        PointGenerator.write_points_to_csv(file_path, points)

    @staticmethod
    def generate_clustered_points_csv(file_path, n, clusters=5, limit=200):
        logging.info(f"Gerando pontos agrupados e salvando no arquivo CSV: {file_path}")
        points = PointGenerator.generate_clustered_points(n, clusters, limit)
        PointGenerator.write_points_to_csv(file_path, points)

    @staticmethod
    def generate_uniform_points_csv(file_path, n, limit=200):
        logging.info(f"Gerando pontos uniformes e salvando no arquivo CSV: {file_path}")
        points = PointGenerator.generate_uniform_points(n, limit)
        PointGenerator.write_points_to_csv(file_path, points)
