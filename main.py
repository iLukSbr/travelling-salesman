import os
from utils import Benchmark

class Main:
    @staticmethod
    def main():
        # Configuração inicial
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = Benchmark.setup_data_directory(base_dir)

        # Definição dos tamanhos de entrada para os benchmarks
        # step_sizes = [100, 500, 1000, 5000, 10000]
        step_sizes = [5]

        # Execução dos benchmarks
        all_results = Benchmark.execute_benchmarks(csv_path, step_sizes)

        # Geração dos gráficos de resultados
        Benchmark.generate_results_plots(all_results)

if __name__ == "__main__":
    Main.main()
