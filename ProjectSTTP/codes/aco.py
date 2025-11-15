# aco_routing/aco.py
import random
import networkx as nx

def aco_route(graph, start, end, edge_costs, n_ants=20, n_iterations=50, alpha=1, beta=3, evaporation=0.5):
    pheromone = {edge: 1.0 for edge in graph.edges}

    def prob(from_node, to_node):
        edge = (from_node, to_node)
        if edge not in edge_costs:
            return 0
        tau = pheromone[edge] ** alpha
        eta = (1.0 / (edge_costs[edge] + 0.001)) ** beta
        return tau * eta

    best_path = None
    best_cost = float('inf')

    for _ in range(n_iterations):
        all_paths = []
        for _ in range(n_ants):
            path = [start]
            while path[-1] != end:
                neighbors = list(graph.successors(path[-1]))
                probs = [prob(path[-1], n) for n in neighbors]
                total = sum(probs)
                if total == 0:
                    break
                probs = [p / total for p in probs]
                next_node = random.choices(neighbors, weights=probs)[0]
                path.append(next_node)
            cost = sum(edge_costs.get((path[i], path[i+1]), 9999) for i in range(len(path)-1))
            if path[-1] == end:
                all_paths.append((path, cost))
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
        # Evaporate
        for edge in pheromone:
            pheromone[edge] *= (1 - evaporation)
        # Deposit
        for path, cost in all_paths:
            for i in range(len(path)-1):
                pheromone[(path[i], path[i+1])] += 1.0 / cost
    return best_path
