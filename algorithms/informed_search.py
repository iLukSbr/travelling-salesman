import heapq
from .algorithm import Algorithm
import logging

# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class InformedSearch(Algorithm):
    def solve(self):
        """
        Resolve o problema do TSP usando busca informada (A*).
        """
        logging.info("Iniciando busca informada (A*)...")
        start_city = 0
        best_route = None
        best_cost = float('inf')
        convergence = []

        # Fila de prioridade para estados (custo estimado, custo atual, rota atual)
        priority_queue = []
        heapq.heappush(priority_queue, (0, 0, [start_city]))

        while priority_queue:
            estimated_cost, current_cost, route = heapq.heappop(priority_queue)

            # Verifica se a rota está completa
            if len(route) == self.n:
                total_cost = current_cost + self.dist_matrix[route[-1]][route[0]]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route + [route[0]]
                convergence.append(best_cost)
                continue

            # Explora as cidades não visitadas
            for next_city in range(self.n):
                if next_city not in route:
                    new_cost = current_cost + self.dist_matrix[route[-1]][next_city]
                    heuristic = self._heuristic(next_city, route)
                    estimated_total_cost = new_cost + heuristic
                    heapq.heappush(priority_queue, (estimated_total_cost, new_cost, route + [next_city]))

        logging.info(f"Busca informada concluída. Menor custo encontrado: {best_cost:.2f}")
        return {
            "route": best_route,
            "cost": best_cost,
            "execution_time": 0,  # Pode ser ajustado para medir o tempo
            "memory_usage": 0,    # Pode ser ajustado para medir o uso de memória
            "convergence": convergence
        }

    def _heuristic(self, current_city, route):
        """
        Heurística para estimar o custo restante (menor aresta para cidades não visitadas).

        Args:
            current_city (int): Cidade atual.
            route (list): Rota atual.

        Returns:
            float: Estimativa do custo restante.
        """
        unvisited = set(range(self.n)) - set(route)
        if not unvisited:
            return self.dist_matrix[current_city][route[0]]  # Retorna ao ponto inicial
        return min(self.dist_matrix[current_city][city] for city in unvisited)
