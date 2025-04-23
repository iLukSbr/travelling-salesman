import os
import matplotlib.pyplot as plt
from .point_generator import PointGenerator
from algorithms import SimulatedAnnealing, Genetic

class Benchmark:
    @staticmethod
    def run_benchmarks(algorithm_instance, step_sizes):
        """
        Executa benchmarks para uma instância de algoritmo.
        """
        results = []
        for step_size in step_sizes:
            print(f"[INFO] Executando benchmark com step size: {step_size}...")
            result = algorithm_instance.solve()
            results.append({
                "step_size": step_size,
                "result": result
            })
        return results

    @staticmethod
    def setup_data_directory(base_dir, file_name="tsp_points.csv", n_points=10, limit=1000):
        """
        Configura o diretório de dados e gera o arquivo CSV com pontos, se necessário.
        """
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        csv_path = os.path.join(data_dir, file_name)

        if not os.path.exists(csv_path):
            print("[INFO] Gerando arquivo CSV com pontos aleatórios...")
            PointGenerator.generate_random_points_csv(csv_path, n=n_points, limit=limit)
            print(f"[INFO] Arquivo CSV gerado em: {csv_path}")
        return csv_path

    @staticmethod
    def load_data(csv_path):
        """
        Carrega os pontos do arquivo CSV.
        """
        print("[INFO] Carregando pontos do arquivo CSV...")
        points = PointGenerator.load_points_from_csv(csv_path)
        print(f"[INFO] {len(points)} pontos carregados.")
        return points

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
    def execute_benchmarks(csv_path, step_sizes):
        """
        Executa benchmarks para os algoritmos e retorna os resultados.
        """
        algorithms = {
            "Têmpera Simulada": SimulatedAnnealing,
            "Algoritmo Genético": Genetic
        }

        points = Benchmark.load_data(csv_path)

        all_results = {}
        for algo_name, algo_class in algorithms.items():
            print(f"[INFO] Executando benchmarks para {algo_name}...")

            # Cria uma instância do algoritmo com os parâmetros necessários
            algo_instance = algo_class(points, **Benchmark.get_algorithm_params(algo_name))

            results = Benchmark.run_benchmarks(
                algo_instance,  # Passa a instância do algoritmo
                step_sizes
            )
            all_results[algo_name] = results
        return all_results

    @staticmethod
    def generate_results_plots(all_results):
        """
        Gera gráficos de desempenho com base nos resultados dos benchmarks.
        """
        print("[INFO] Gerando gráficos de resultados...")
        Benchmark.plot_comparative_results(all_results, output_dir="benchmark_results")

    def plot_comparative_results(all_results, output_dir="benchmark_results"):
        """
        Gera gráficos comparativos entre os algoritmos com base em tempo, memória e convergência.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Gráfico de tempo de execução e uso de memória
        algorithms = list(all_results.keys())
        step_sizes = [result["step_size"] for result in all_results[algorithms[0]]]
        execution_times = {algo: [result["result"]["execution_time"] for result in results] for algo, results in
                           all_results.items()}
        memory_usages = {algo: [result["result"]["memory_usage"] for result in results] for algo, results in
                         all_results.items()}

        # Gráfico comparativo de tempo de execução e uso de memória
        plt.figure(figsize=(12, 6))
        for algo in algorithms:
            plt.plot(step_sizes, execution_times[algo], marker="o", label=f"{algo} - Tempo de Execução")
        plt.title("Comparação de Tempo de Execução")
        plt.xlabel("Step Size")
        plt.ylabel("Tempo de Execução (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "execution_time_comparison.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        for algo in algorithms:
            plt.plot(step_sizes, memory_usages[algo], marker="o", label=f"{algo} - Uso de Memória")
        plt.title("Comparação de Uso de Memória")
        plt.xlabel("Step Size")
        plt.ylabel("Uso de Memória (MB)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "memory_usage_comparison.png"))
        plt.close()

        # Gráfico comparativo de convergência
        plt.figure(figsize=(12, 6))
        for algo in algorithms:
            for step_size in step_sizes:
                convergence = next(
                    result["result"]["convergence"] for result in all_results[algo] if result["step_size"] == step_size)
                plt.plot(range(len(convergence)), convergence, label=f"{algo} - Step Size {step_size}")
        plt.title("Comparação de Convergência")
        plt.xlabel("Iterações")
        plt.ylabel("Custo")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "convergence_comparison.png"))
        plt.close()

        print(f"[INFO] Gráficos comparativos salvos em: {output_dir}")
