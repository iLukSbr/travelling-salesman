import os
import logging
from utils import Benchmark

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class Main:
    @staticmethod
    def main():
        # Configuração inicial
        base_dir = os.path.dirname(os.path.abspath(__file__))
        point_quantities = [50, 100, 200, 500, 1000, 2000]  # Quantidades variadas de pontos
        step_sizes = [1, 10, 50] # Tamanhos de passo para os algoritmos

        # Gera arquivos CSV para diferentes quantidades de pontos
        logging.info("Configurando diretórios de dados e gerando arquivos CSV...")
        csv_paths = Benchmark.setup_data_directories(base_dir, point_quantities)

        # Executa benchmarks e gera gráficos para cada quantidade de pontos
        for n_points, csv_path in csv_paths.items():
            logging.info(f"Executando benchmarks para {n_points} pontos...")
            Benchmark.execute_and_plot(csv_path, n_points, step_sizes, output_dir=f"benchmark_results/{n_points}_points")

if __name__ == "__main__":
    Main.main()
