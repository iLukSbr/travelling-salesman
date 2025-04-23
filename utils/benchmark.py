import time
import psutil
import matplotlib.pyplot as plt
from utils.point_generator import load_points_from_csv
import os

class Benchmark:
    @staticmethod
    def measure_resources(func, *args, **kwargs):
        """
        Mede o tempo de execução, uso de CPU e memória de uma função.

        Args:
            func (callable): Função a ser executada.
            *args: Argumentos posicionais para a função.
            **kwargs: Argumentos nomeados para a função.

        Returns:
            tuple: Resultado da função e métricas de desempenho.
        """
        cpu_start = psutil.cpu_percent(interval=None)
        memory_start = psutil.virtual_memory().used / (1024 ** 2)  # Em MB
        start_time = time.perf_counter()

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        cpu_end = psutil.cpu_percent(interval=None)
        memory_end = psutil.virtual_memory().used / (1024 ** 2)  # Em MB

        metrics = {
            "time": end_time - start_time,
            "cpu_usage": cpu_end - cpu_start,
            "memory_usage": memory_end - memory_start
        }
        return result, metrics

    @staticmethod
    def benchmark_algorithm(algorithm, cities, **kwargs):
        """
        Executa o algoritmo e mede tempo, custo e uso de recursos do sistema.

        Args:
            algorithm (function or class): Função ou classe do algoritmo a ser executado.
            cities (list): Lista de coordenadas das cidades.
            **kwargs: Parâmetros adicionais para o algoritmo.

        Returns:
            dict: Resultados contendo melhor rota, custo, tempo de execução e uso de recursos.
        """
        if callable(algorithm):
            func = algorithm
        else:
            func = lambda cities, **kwargs: algorithm(cities, **kwargs).solve()

        (best_route, best_cost), metrics = Benchmark.measure_resources(func, cities, **kwargs)
        return {
            "route": best_route,
            "cost": best_cost,
            **metrics
        }

    @staticmethod
    def run_benchmarks(algorithms, csv_path, step_sizes, **kwargs):
        """
        Executa benchmarks para diferentes algoritmos e tamanhos de entrada.

        Args:
            algorithms (dict): Dicionário com nomes e funções/classes dos algoritmos.
            csv_path (str): Caminho para o arquivo CSV com os dados.
            step_sizes (list): Lista de tamanhos de entrada para os benchmarks.
            **kwargs: Parâmetros adicionais para os algoritmos.

        Returns:
            dict: Resultados contendo custo, tempo, uso de recursos e tamanho da entrada para cada algoritmo.
        """
        points = load_points_from_csv(csv_path)
        max_points = len(points)
        all_results = {}

        for algo_name, algorithm in algorithms.items():
            print(f"[INFO] Executando benchmarks para o algoritmo: {algo_name}")
            results = []

            for size in step_sizes:
                if size > max_points:
                    print(f"Tamanho {size} excede o número máximo de pontos ({max_points}). Ignorando.")
                    continue

                print(f"Executando benchmark para {size} pontos...")
                subset = points[:size]
                result = Benchmark.benchmark_algorithm(algorithm, subset, **kwargs)
                result["size"] = size
                results.append(result)

                print(f"Concluído: {size} pontos - Custo: {result['cost']:.6f}, "
                      f"Tempo: {result['time']:.6f}s, CPU: {result['cpu_usage']:.2f}%, "
                      f"Memória: {result['memory_usage']:.2f}MB")

            all_results[algo_name] = results

        return all_results

    @staticmethod
    def plot_benchmark_results(all_results, output_dir="benchmark_results"):
        """
        Gera gráficos de desempenho com base nos resultados do benchmark e salva em arquivos.

        Args:
            all_results (dict): Resultados do benchmark para cada algoritmo.
            output_dir (str): Diretório onde os gráficos serão salvos.
        """
        os.makedirs(output_dir, exist_ok=True)

        for algo_name, results in all_results.items():
            sizes = [result["size"] for result in results]
            metrics = {
                "Tempo de Execução (s)": [result["time"] for result in results],
                "Custo da Solução": [result["cost"] for result in results],
                "Uso de CPU (%)": [result["cpu_usage"] for result in results],
                "Uso de Memória (MB)": [result["memory_usage"] for result in results]
            }

            for metric_name, values in metrics.items():
                plt.figure(figsize=(12, 6))
                plt.plot(sizes, values, marker='o', label=metric_name)
                plt.title(f"{metric_name} - {algo_name}")
                plt.xlabel("Tamanho da Entrada (número de pontos)")
                plt.ylabel(metric_name)
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(output_dir, f"{algo_name}_{metric_name.replace(' ', '_').lower()}.png"))
                plt.close()
