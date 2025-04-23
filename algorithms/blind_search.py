import random
from collections import deque
from .algorithm import Algorithm
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class BlindSearch(Algorithm):
    def solve(self):
        """
        Resolve o problema do TSP usando uma busca cega que mistura BFS e DFS.
        """
        logging.info("Iniciando busca cega (Blind Search)...")
        start_city = random.randint(0, self.n - 1)
        best_route = None
        best_cost = float('inf')
        convergence = []

        # Inicializa a pilha (DFS) e a fila (BFS)
        stack = [(start_city, [start_city], 0)]  # (cidade_atual, rota, custo_atual)
        queue = deque([(start_city, [start_city], 0)])

        while stack or queue:
            # Alterna entre DFS e BFS
            if random.random() < 0.5 and stack:
                current_city, route, current_cost = stack.pop()  # DFS
            elif queue:
                current_city, route, current_cost = queue.popleft()  # BFS
            else:
                continue

            # Verifica se a rota está completa
            if len(route) == self.n:
                total_cost = current_cost + self.dist_matrix[current_city][route[0]]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route + [route[0]]
                convergence.append(best_cost)
                continue

            # Explora as cidades não visitadas
            for next_city in range(self.n):
                if next_city not in route:
                    new_cost = current_cost + self.dist_matrix[current_city][next_city]
                    new_route = route + [next_city]
                    stack.append((next_city, new_route, new_cost))
                    queue.append((next_city, new_route, new_cost))

        logging.info(f"Busca cega concluída. Menor custo encontrado: {best_cost:.2f}")
        return {
            "route": best_route,
            "cost": best_cost,
            "execution_time": 0,  # Pode ser ajustado para medir o tempo
            "memory_usage": 0,    # Pode ser ajustado para medir o uso de memória
            "convergence": convergence
        }