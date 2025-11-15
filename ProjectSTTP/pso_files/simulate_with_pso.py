# import csv
# import random
# import xml.etree.ElementTree as ET
# from collections import defaultdict, deque

# def parse_edge_list(value):
#     return [e.strip() for e in value.split(',') if e.strip()] if value else []

# def build_junction_graph(csv_path):
#     edge_to_junction = {}
#     junction_graph = defaultdict(set)
#     incoming_edges_map = {}
#     outgoing_edges_map = {}

#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             junction = row['junction_id'].strip()
#             incoming_edges = parse_edge_list(row.get('incoming_edges', ''))
#             outgoing_edges = parse_edge_list(row.get('outgoing_edges', ''))

#             incoming_edges_map[junction] = incoming_edges
#             outgoing_edges_map[junction] = outgoing_edges

#             for edge in incoming_edges + outgoing_edges:
#                 edge_to_junction[edge.lstrip('-')] = junction

#     for j1 in incoming_edges_map:
#         for out_edge in outgoing_edges_map.get(j1, []):
#             clean = out_edge.lstrip('-')
#             for j2, in_edges in incoming_edges_map.items():
#                 if clean in [e.lstrip('-') for e in in_edges] and j1 != j2:
#                     junction_graph[j1].add(j2)

#     return junction_graph, edge_to_junction

# def load_distance_matrix(path):
#     distance = {}
#     with open(path, 'r') as f:
#         reader = csv.reader(f)
#         headers = list(map(int, next(reader)[1:]))
#         for row in reader:
#             from_j = int(row[0])
#             distance[from_j] = {headers[i]: float(val) if val else float('inf') for i, val in enumerate(row[1:])}
#     return distance

# def load_density_at_time(csv_path, time_step):
#     densities = {}
#     with open(csv_path, 'r') as f:
#         reader = csv.reader(f)
#         times = list(map(int, next(reader)[1:]))
#         if time_step not in times:
#             raise ValueError("Time step not found in CSV")
#         idx = times.index(time_step)
#         for row in reader:
#             junction = int(row[0])
#             densities[junction] = float(row[1 + idx])
#     return densities

# def build_weighted_graph(junction_graph, distances, densities, alpha=1.0):
#     weighted = {}
#     for j1 in junction_graph:
#         weighted[j1] = {}
#         for j2 in junction_graph[j1]:
#             d = distances.get(int(j1), {}).get(int(j2), float('inf'))
#             den = densities.get(int(j2), 0.0)
#             cost = d + alpha * den
#             if cost < float('inf'):
#                 weighted[j1][j2] = cost
#     return weighted

# def run_pso(weighted_graph, source, dest, num_particles=20, num_iterations=50):
#     def fitness(path):
#         return sum(weighted_graph[path[i]].get(path[i+1], float('inf')) for i in range(len(path)-1))

#     def generate_path():
#         for _ in range(50):
#             path = [source]
#             visited = set(path)
#             while path[-1] != dest:
#                 current = path[-1]
#                 neighbors = list(weighted_graph.get(current, {}))
#                 neighbors = [n for n in neighbors if n not in visited]
#                 if not neighbors:
#                     break
#                 next_node = random.choice(neighbors)
#                 path.append(next_node)
#                 visited.add(next_node)
#             if path[-1] == dest:
#                 return path
#         return None

#     particles = [generate_path() for _ in range(num_particles)]
#     particles = [p for p in particles if p]
#     if not particles:
#         return None, float('inf')

#     g_best = min(particles, key=fitness)
#     g_best_cost = fitness(g_best)

#     for _ in range(num_iterations):
#         for i in range(len(particles)):
#             candidate = generate_path()
#             if candidate and fitness(candidate) < fitness(particles[i]):
#                 particles[i] = candidate
#         current_best = min(particles, key=fitness)
#         current_cost = fitness(current_best)
#         if current_cost < g_best_cost:
#             g_best = current_best
#             g_best_cost = current_cost

#     return g_best, g_best_cost

# def extract_vehicle_routes_from_xml(xml_path):
#     vehicle_data = []
#     root = ET.parse(xml_path).getroot()
#     for vehicle in root.findall('vehicle'):
#         vid = vehicle.attrib['id']
#         depart = float(vehicle.attrib.get('depart', 0))
#         route = vehicle.find('route')
#         if route is not None:
#             edges = route.attrib['edges'].strip().split()
#             if len(edges) >= 2:
#                 source = edges[0].lstrip('-')
#                 dest = edges[-1].lstrip('-')
#                 vehicle_data.append((vid, source, dest, depart))
#     return vehicle_data

# def path_exists(graph, source, dest):
#     visited = set()
#     queue = deque([source])
#     while queue:
#         node = queue.popleft()
#         if node == dest:
#             return True
#         for nbr in graph.get(node, []):
#             if nbr not in visited:
#                 visited.add(nbr)
#                 queue.append(nbr)
#     return False

# def write_final_routes(final_routes, output_path):
#     root = ET.Element("routes", {
#         "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
#         "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
#     })
#     for vid, route, depart in final_routes:
#         v_elem = ET.SubElement(root, "vehicle", {
#             "id": vid,
#             "type": "slow_vehicle",
#             "depart": f"{depart:.2f}"
#         })
#         ET.SubElement(v_elem, "route", {"edges": ' '.join(route)})
#     tree = ET.ElementTree(root)
#     tree.write(output_path, encoding='utf-8', xml_declaration=True)

# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_csv = "../models/junction_io_edges.csv"
#     distance_file = "../data/aligned_normalized_distance_matrix.csv"
#     density_file = "../simulation_data/runtime_routes.csv"
#     output_csv = "pso_rerouted_paths.csv"
#     output_xml = "pso_final_edge_route.xml"

#     junction_graph, edge_to_junction = build_junction_graph(junction_csv)
#     distances = load_distance_matrix(distance_file)
#     vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

#     final_routes = []
#     with open(output_csv, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['timestamp', 'vehicle_id', 'route'])

#         for t in range(0, 14400, 300):
#             next_t = t + 300
#             try:
#                 densities = load_density_at_time(density_file, next_t)
#             except ValueError:
#                 break
#             weighted_graph = build_weighted_graph(junction_graph, distances, densities)

#             current_result = []
#             for vehicle_id, source_edge, dest_edge, depart in vehicle_routes:
#                 source_j = edge_to_junction.get(source_edge)
#                 dest_j = edge_to_junction.get(dest_edge)

#                 if not source_j or not dest_j:
#                     continue
#                 if not path_exists(junction_graph, source_j, dest_j):
#                     continue

#                 path, cost = run_pso(weighted_graph, source_j, dest_j)
#                 if path:
#                     writer.writerow([t, vehicle_id, ' '.join(path)])
#                     current_result.append((vehicle_id, path, depart))
#             if current_result:
#                 final_routes = current_result

#     write_final_routes(final_routes, output_xml)

# if __name__ == "__main__":
#     main()
# Let's correct the PSO routing program to convert junction paths into valid edge-based routes.
# We'll define the required code structure to match the ACO-style output with edge-based routing.

import csv
import random
from collections import defaultdict, deque
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def parse_edge_list(value):
    return [e.strip() for e in value.split(',') if e.strip()]

def build_junction_graph(csv_path):
    edge_to_junction = {}
    junction_graph = defaultdict(set)
    all_junctions = set()
    incoming_edges_map = {}
    outgoing_edges_map = {}
    edge_map = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            junction = row['junction_id'].strip()
            all_junctions.add(junction)

            incoming_edges = parse_edge_list(row.get('incoming_edges', ''))
            outgoing_edges = parse_edge_list(row.get('outgoing_edges', ''))

            incoming_edges_map[junction] = incoming_edges
            outgoing_edges_map[junction] = outgoing_edges

            for edge in incoming_edges + outgoing_edges:
                edge_clean = edge.lstrip('-')
                edge_to_junction[edge_clean] = junction

    for j1 in all_junctions:
        for out_edge in outgoing_edges_map.get(j1, []):
            clean = out_edge.lstrip('-')
            for j2, in_edges in incoming_edges_map.items():
                if clean in [e.lstrip('-') for e in in_edges] and j1 != j2:
                    junction_graph[j1].add(j2)
                    edge_map[(j1, j2)] = out_edge

    return junction_graph, edge_to_junction, edge_map

def load_distance_matrix(path):
    distance = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = list(map(int, next(reader)[1:]))
        for row in reader:
            from_j = int(row[0])
            distance[from_j] = {headers[i]: float(val) if val else float('inf') for i, val in enumerate(row[1:])}
    return distance

def load_density_at_time(csv_path, time_step):
    densities = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        times = list(map(int, next(reader)[1:]))
        idx = times.index(time_step)
        for row in reader:
            junction = int(row[0])
            densities[junction] = float(row[1 + idx])
    return densities

def build_weighted_graph(junction_graph, distances, densities, alpha=1.0):
    weighted = {}
    for j1 in junction_graph:
        weighted[j1] = {}
        for j2 in junction_graph[j1]:
            d = distances.get(int(j1), {}).get(int(j2), float('inf'))
            den = densities.get(int(j2), 0.0)
            cost = d + alpha * den
            weighted[j1][j2] = cost
    return weighted

def run_pso(weighted_graph, source, dest, num_particles=20, num_iterations=50):
    def fitness(path):
        return sum(weighted_graph[path[i]].get(path[i+1], float('inf')) for i in range(len(path)-1))

    def generate_path():
        for _ in range(50):
            path = [source]
            visited = set(path)
            while path[-1] != dest:
                current = path[-1]
                neighbors = [n for n in weighted_graph.get(current, {}) if n not in visited]
                if not neighbors:
                    break
                path.append(random.choice(neighbors))
                visited.add(path[-1])
            if path[-1] == dest:
                return path
        return None

    particles = [generate_path() for _ in range(num_particles)]
    particles = [p for p in particles if p]
    if not particles:
        return None, float('inf')

    g_best = min(particles, key=fitness)
    g_best_cost = fitness(g_best)

    for _ in range(num_iterations):
        for i in range(len(particles)):
            candidate = generate_path()
            if candidate and fitness(candidate) < fitness(particles[i]):
                particles[i] = candidate
        current_best = min(particles, key=fitness)
        if fitness(current_best) < g_best_cost:
            g_best, g_best_cost = current_best, fitness(current_best)

    return g_best, g_best_cost

def extract_vehicle_routes_from_xml(xml_path):
    vehicle_data = []
    root = ET.parse(xml_path).getroot()
    for vehicle in root.findall('vehicle'):
        vid = vehicle.attrib['id']
        depart = float(vehicle.attrib.get('depart', 0))
        route = vehicle.find('route')
        if route is not None:
            edges = route.attrib['edges'].strip().split()
            if len(edges) >= 2:
                source = edges[0].lstrip('-')
                dest = edges[-1].lstrip('-')
                vehicle_data.append((vid, source, dest, depart))
    return vehicle_data

def path_exists(graph, source, dest):
    visited = set()
    queue = deque([source])
    while queue:
        node = queue.popleft()
        if node == dest:
            return True
        for nbr in graph.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
    return False

def convert_junction_path_to_edge_route(junction_path, edge_map):
    edge_route = []
    for i in range(len(junction_path) - 1):
        j_from, j_to = junction_path[i], junction_path[i + 1]
        edge = edge_map.get((j_from, j_to))
        if not edge:
            return None
        edge_route.append(edge)
    return edge_route

def write_pretty_xml(final_routes, output_path="pso_final_edge_route.xml"):
    root = ET.Element("routes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
    })

    for vid, route, depart in final_routes:
        v_elem = ET.SubElement(root, "vehicle", {
            "id": vid,
            "type": "slow_vehicle",
            "depart": f"{depart:.2f}"
        })
        ET.SubElement(v_elem, "route", {"edges": ' '.join(route)})

    rough_string = ET.tostring(root, 'utf-8')
    pretty_xml = minidom.parseString(rough_string).toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml)

# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_csv = "../models/junction_io_edges.csv"
#     distance_file = "../data/aligned_normalized_distance_matrix.csv"
#     density_file = "../simulation_data/runtime_routes.csv"
#     output_csv = "pso_rerouted_paths.csv"
#     output_xml = "pso_final_edge_route.xml"

#     junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
#     distances = load_distance_matrix(distance_file)
#     vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

#     final_routes = []
#     with open(output_csv, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['timestamp', 'vehicle_id', 'route'])

#         for t in range(0, 14400, 300):
#             next_t = t + 300
#             try:
#                 densities = load_density_at_time(density_file, next_t)
#             except ValueError:
#                 break
#             weighted_graph = build_weighted_graph(junction_graph, distances, densities)

#             current_result = []
#             for vid, src_edge, dst_edge, depart in vehicle_routes:
#                 src_junc = edge_to_junction.get(src_edge)
#                 dst_junc = edge_to_junction.get(dst_edge)

#                 if not src_junc or not dst_junc:
#                     continue
#                 if not path_exists(junction_graph, src_junc, dst_junc):
#                     continue

#                 junction_path, cost = run_pso(weighted_graph, src_junc, dst_junc)
#                 if junction_path:
#                     edge_route = convert_junction_path_to_edge_route(junction_path, edge_map)
#                     if edge_route:
#                         writer.writerow([t, vid, ' '.join(edge_route)])
#                         current_result.append((vid, edge_route, depart))
#             if current_result:
#                 final_routes = current_result

#     write_pretty_xml(final_routes, output_xml)
#     print(f"✅ Final PSO edge routes saved to: {output_xml}")


def main():
    xml_path = "../routes/aco_vehicle.xml"
    junction_csv = "../models/junction_io_edges.csv"
    distance_file = "../data/aligned_normalized_distance_matrix.csv"
    density_file = "../simulation_data/runtime_routes.csv"

    # Parameter combinations: (num_particles, num_iterations)
    param_combinations = [
        (10, 20), (10, 40), (10, 60), (10, 80), (10, 100),
        (20, 20), (20, 40), (20, 60), (20, 80), (20, 100)
    ]

    junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
    distances = load_distance_matrix(distance_file)
    vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

    for trial, (num_particles, num_iterations) in enumerate(param_combinations, 1):
        print(f"\n🚀 Running Trial {trial}: Particles={num_particles}, Iterations={num_iterations}")
        output_csv = f"pso_rerouted_paths_trial{trial}.csv"
        output_xml = f"pso_final_edge_route_trial{trial}.xml"
        final_routes = []

        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'vehicle_id', 'route'])

            for t in range(0, 14400, 300):
                next_t = t + 300
                try:
                    densities = load_density_at_time(density_file, next_t)
                except ValueError:
                    break
                weighted_graph = build_weighted_graph(junction_graph, distances, densities)

                current_result = []
                for vid, src_edge, dst_edge, depart in vehicle_routes:
                    src_junc = edge_to_junction.get(src_edge)
                    dst_junc = edge_to_junction.get(dst_edge)

                    if not src_junc or not dst_junc:
                        continue
                    if not path_exists(junction_graph, src_junc, dst_junc):
                        continue

                    junction_path, cost = run_pso(weighted_graph, src_junc, dst_junc,
                                                  num_particles=num_particles,
                                                  num_iterations=num_iterations)
                    if junction_path:
                        edge_route = convert_junction_path_to_edge_route(junction_path, edge_map)
                        if edge_route:
                            writer.writerow([t, vid, ' '.join(edge_route)])
                            current_result.append((vid, edge_route, depart))

                if current_result:
                    final_routes = current_result  # overwrite with latest timestamp routes

        write_pretty_xml(final_routes, output_xml)
        print(f"✅ Trial {trial} completed. Output saved: {output_xml}")


if __name__ == "__main__":
    main()
