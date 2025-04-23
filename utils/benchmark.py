import logging
import os
import matplotlib.pyplot as plt
from .point_generator import PointGenerator
from algorithms import SimulatedAnnealing, Genetic

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class Benchmark:
    @staticmethod
    def run_and_collect(algorithm_instance, step_sizes):
        """
        Executa benchmarks para uma instância de algoritmo e coleta os resultados.
        """
        results = []
        for step_size in step_sizes:
            logging.info(f"Executando benchmark com step size: {step_size}...")
            result = algorithm_instance.solve()
            results.append({
                "step_size": step_size,
                "result": {
                    "route": result["route"],
                    "cost": result["cost"],
                    "execution_time": result["execution_time"],
                    "memory_usage": result["memory_usage"],
                    "convergence": result["convergence"]
                }
            })
        return results

    @staticmethod
    def setup_data_directories(base_dir, point_quantities, limit=1000):
        """
        Configura o diretório de dados e gera arquivos CSV para diferentes quantidades de pontos.
        """
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        csv_paths = {}
        for n_points in point_quantities:
            file_name = f"tsp_points_{n_points}.csv"
            csv_path = os.path.join(data_dir, file_name)

            if not os.path.exists(csv_path):
                logging.info(f"Gerando arquivo CSV com {n_points} pontos aleatórios...")
                PointGenerator.generate_random_points_csv(csv_path, n=n_points, limit=limit)
                logging.info(f"Arquivo CSV gerado em: {csv_path}")

            csv_paths[n_points] = csv_path
        return csv_paths

    @staticmethod
    def load_data(csv_path):
        """
        Carrega os pontos do arquivo CSV.
        """
        logging.info(f"Carregando pontos do arquivo: {csv_path}...")
        points = PointGenerator.load_points_from_csv(csv_path)
        logging.info(f"{len(points)} pontos carregados.")
        return points

    @staticmethod
    def get_algorithm_params(algorithm_name):
        """
        Retorna os parâmetros específicos para cada algoritmo.
        """
        params = {
            "Têmpera Simulada": {
                "initial_temperature": 1000,
                "cooling_rate": 0.95,
                "max_iterations": 1000
            },
            "Algoritmo Genético": {
                "pop_size": 100,
                "generations": 500,
                "mutation_rate": 0.02,
                "elitism_size": 5,
                "tournament_size": 5
            }
        }
        return params.get(algorithm_name, {})

    @staticmethod
    def execute_and_plot(csv_path, n_points, step_sizes, output_dir="benchmark_results"):
        """
        Executa benchmarks para os algoritmos e gera gráficos diretamente.
        """
        algorithms = {
            "Têmpera Simulada": SimulatedAnnealing,
            "Algoritmo Genético": Genetic
        }

        points = Benchmark.load_data(csv_path)
        output_dir = f"{output_dir}/{n_points}_points"  # Adiciona o sufixo "_points" diretamente
        os.makedirs(output_dir, exist_ok=True)

        for algo_name, algo_class in algorithms.items():
            logging.info(f"Executando benchmarks para {algo_name} com {n_points} pontos...")
            algo_instance = algo_class(points, **Benchmark.get_algorithm_params(algo_name))
            results = Benchmark.run_and_collect(algo_instance, step_sizes)

            # Gera gráficos para o algoritmo atual
            step_sizes = [result["step_size"] for result in results]
            execution_times = [result["result"]["execution_time"] for result in results]
            memory_usages = [result["result"]["memory_usage"] for result in results]

            # Gráfico de tempo de execução
            plt.figure(figsize=(12, 6))
            plt.plot(step_sizes, execution_times, marker="o", label=f"{algo_name}")
            plt.title(f"Tempo de Execução - {algo_name} ({n_points} pontos)")
            plt.xlabel("Step Size")
            plt.ylabel("Tempo de Execução (s)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{algo_name}_execution_time_{n_points}.png"))
            plt.close()

            # Gráfico de uso de memória
            plt.figure(figsize=(12, 6))
            plt.plot(step_sizes, memory_usages, marker="o", label=f"{algo_name}")
            plt.title(f"Uso de Memória - {algo_name} ({n_points} pontos)")
            plt.xlabel("Step Size")
            plt.ylabel("Uso de Memória (MB)")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{algo_name}_memory_usage_{n_points}.png"))
            plt.close()

        logging.info(f"Gráficos salvos em: {output_dir}")