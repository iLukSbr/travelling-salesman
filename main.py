import os
from utils import *
from algorithms import *

class Main:
    @staticmethod
    def setup_data_directory(base_dir, file_name="tsp_points.csv", n_points=10000, limit=1000):
        """
        Configura o diretório de dados e gera o arquivo CSV com pontos, se necessário.
        """
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, file_name)

        if not os.path.exists(csv_path):
            print("[INFO] Gerando arquivo CSV com pontos aleatórios...")
            generate_random_points_csv(csv_path, n=n_points, limit=limit)
            print(f"[INFO] Arquivo CSV gerado em: {csv_path}")
        return csv_path

    @staticmethod
    def load_data(csv_path):
        """
        Carrega os pontos do arquivo CSV.
        """
        print("[INFO] Carregando pontos do arquivo CSV...")
        points = load_points_from_csv(csv_path)
        print(f"[INFO] {len(points)} pontos carregados.")
        return points

    @staticmethod
    def execute_benchmarks(csv_path, step_sizes):
        """
        Executa benchmarks para os algoritmos e retorna os resultados.
        """
        algorithms = {
            "Têmpera Simulada": simulated_annealing,
            "Algoritmo Genético": GeneticAlgorithm
        }

        all_results = {}
        for algo_name, algo_func in algorithms.items():
            print(f"[INFO] Executando benchmarks para {algo_name}...")
            results = Benchmark.run_benchmarks(
                {algo_name: algo_func},
                csv_path,
                step_sizes,
                **Main.get_algorithm_params(algo_name)
            )
            all_results.update(results)
        return all_results

    @staticmethod
    def get_algorithm_params(algorithm_name):
        """
        Retorna os parâmetros específicos para cada algoritmo.
        """
        if algorithm_name == "Têmpera Simulada":
            return {
                "initial_temperature": 1000,
                "cooling_rate": 0.95,
                "max_iterations": 1000
            }
        elif algorithm_name == "Algoritmo Genético":
            return {
                "pop_size": 100,
                "generations": 500,
                "mutation_rate": 0.02,
                "elitism_size": 5,
                "tournament_size": 5
            }
        return {}

    @staticmethod
    def generate_results_plots(all_results):
        """
        Gera gráficos de desempenho com base nos resultados dos benchmarks.
        """
        print("[INFO] Gerando gráficos de resultados...")
        Benchmark.plot_benchmark_results(all_results, output_dir="benchmark_results")

    @staticmethod
    def main():
        # Configuração inicial
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = Main.setup_data_directory(base_dir)

        # Carregamento de dados
        points = Main.load_data(csv_path)

        # Definição dos tamanhos de entrada para os benchmarks
        step_sizes = [100, 500, 1000, 5000, 10000, 100000, 1000000]

        # Execução dos benchmarks
        all_results = Main.execute_benchmarks(csv_path, step_sizes)

        # Geração dos gráficos de resultados
        Main.generate_results_plots(all_results)

if __name__ == "__main__":
    Main.main()
