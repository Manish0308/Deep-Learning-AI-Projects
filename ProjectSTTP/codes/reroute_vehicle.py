# aco_routing/reroute_vehicle.py
from aco import aco_route
from build_graph import build_graph
from edge_cost import get_edge_costs_from_density

def reroute_vehicle(netfile, density_file, start_node, end_node):
    G = build_graph(netfile)
    edge_costs = get_edge_costs_from_density(netfile, density_file)
    # Convert node-based to edge-based cost
    edge_costs = {(u, v): edge_costs[G[u][v]['id']] for u, v in G.edges if G[u][v]['id'] in edge_costs}
    best_path = aco_route(G, start_node, end_node, edge_costs)
    print("Best path:", best_path)
    return best_path
