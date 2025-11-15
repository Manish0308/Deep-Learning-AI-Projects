# import csv
# import random
# import xml.etree.ElementTree as ET
# from collections import defaultdict, deque

# def parse_edge_list(value):
#     if not value:
#         return []
#     return [e.strip() for e in value.split(',') if e.strip()]

# def build_junction_graph(csv_path):
#     edge_to_junction = {}
#     junction_graph = defaultdict(set)
#     all_junctions = set()
#     incoming_edges_map = {}
#     outgoing_edges_map = {}
#     edge_map = {}

#     with open(csv_path, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             junction = row['junction_id'].strip()
#             all_junctions.add(junction)
#             incoming_edges = parse_edge_list(row.get('incoming_edges', ''))
#             outgoing_edges = parse_edge_list(row.get('outgoing_edges', ''))
#             incoming_edges_map[junction] = incoming_edges
#             outgoing_edges_map[junction] = outgoing_edges
#             for edge in incoming_edges + outgoing_edges:
#                 edge_clean = edge.lstrip('-')
#                 edge_to_junction[edge_clean] = junction

#     for j1 in all_junctions:
#         for out_edge in outgoing_edges_map.get(j1, []):
#             edge_clean = out_edge.lstrip('-')
#             for j2, in_edges in incoming_edges_map.items():
#                 if edge_clean in [e.lstrip('-') for e in in_edges] and j1 != j2:
#                     junction_graph[j1].add(j2)
#                     edge_map[(j1, j2)] = out_edge
#         for in_edge in incoming_edges_map.get(j1, []):
#             edge_clean = in_edge.lstrip('-')
#             for j0, out_edges in outgoing_edges_map.items():
#                 if edge_clean in [e.lstrip('-') for e in out_edges] and j0 != j1:
#                     junction_graph[j0].add(j1)
#                     edge_map[(j0, j1)] = in_edge

#     for junc in all_junctions:
#         if junc not in junction_graph:
#             junction_graph[junc] = set()

#     return junction_graph, edge_to_junction, edge_map

# def load_distance_matrix(path):
#     distance = {}
#     with open(path, 'r') as f:
#         reader = csv.reader(f)
#         headers = list(map(int, next(reader)[1:]))
#         for row in reader:
#             from_j = int(row[0])
#             distance[from_j] = {}
#             for i, val in enumerate(row[1:]):
#                 try:
#                     distance[from_j][headers[i]] = float(val)
#                 except ValueError:
#                     distance[from_j][headers[i]] = float('inf')
#     return distance

# def load_density_at_time(csv_path, time_step):
#     densities = {}
#     with open(csv_path, 'r') as f:
#         reader = csv.reader(f)
#         times = list(map(int, next(reader)[1:]))
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
#             weighted[j1][j2] = cost
#     return weighted

# def run_aco(weighted_graph, source, dest, num_ants=15, num_iterations=40,
#             alpha=1.0, beta=2.0, evaporation_rate=0.5, q=100.0):
#     pheromone = {u: {v: 1.0 for v in weighted_graph[u]} for u in weighted_graph}
#     best_path = None
#     best_cost = float('inf')
#     for _ in range(num_iterations):
#         for _ in range(num_ants):
#             path = [source]
#             visited = set(path)
#             cost = 0
#             while path[-1] != dest:
#                 current = path[-1]
#                 neighbors = [n for n in weighted_graph.get(current, {}) if n not in visited]
#                 if not neighbors:
#                     break
#                 probs = [(pheromone[current][n] ** alpha) * (1.0 / weighted_graph[current][n] ** beta) for n in neighbors]
#                 total = sum(probs)
#                 if total == 0:
#                     break
#                 probs = [p / total for p in probs]
#                 next_node = random.choices(neighbors, weights=probs)[0]
#                 path.append(next_node)
#                 visited.add(next_node)
#                 cost += weighted_graph[current][next_node]
#             if path[-1] == dest and cost < best_cost:
#                 best_path, best_cost = path, cost
#         for u in pheromone:
#             for v in pheromone[u]:
#                 pheromone[u][v] *= (1 - evaporation_rate)
#         for i in range(len(best_path or []) - 1):
#             u, v = best_path[i], best_path[i + 1]
#             pheromone[u][v] += q / (best_cost + 1e-6)
#     return best_path, best_cost

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

# def extract_vehicle_routes_from_xml(xml_path):
#     vehicle_data = []
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     for vehicle in root.findall('vehicle'):
#         vid = vehicle.attrib['id']
#         depart = float(vehicle.attrib['depart'])
#         route = vehicle.find('route')
#         if route is not None:
#             edges = route.attrib['edges'].strip().split()
#             if len(edges) >= 2:
#                 source_edge = edges[0].lstrip('-')
#                 dest_edge = edges[-1].lstrip('-')
#                 vehicle_data.append((vid, source_edge, dest_edge, depart))
#     return vehicle_data

# def convert_junction_path_to_edge_route(junction_path, edge_map):
#     edge_route = []
#     for i in range(len(junction_path) - 1):
#         edge = edge_map.get((junction_path[i], junction_path[i+1]))
#         if edge:
#             edge_route.append(edge)
#         else:
#             return None
#     return edge_route

# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_csv = "../models/junction_io_edges.csv"
#     distance_file = "../data/aligned_normalized_distance_matrix.csv"
#     density_file = "../simulation_data/runtime_routes.csv"
#     output_csv = "aco_rerouted_paths.csv"
#     output_xml = "aco_final_edge_route.xml"

#     junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
#     distances = load_distance_matrix(distance_file)
#     vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

#     root = ET.Element("routes", {
#         "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
#         "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
#     })

#     with open(output_csv, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['timestamp', 'vehicle_id', 'route'])

#         for t in range(0, 14400, 300):
#             next_t = t + 300
#             try:
#                 densities = load_density_at_time(density_file, next_t)
#             except ValueError:
#                 continue
#             weighted_graph = build_weighted_graph(junction_graph, distances, densities)
#             for vid, src_edge, dst_edge, depart in vehicle_routes:
#                 src_junc = edge_to_junction.get(src_edge)
#                 dst_junc = edge_to_junction.get(dst_edge)
#                 if not src_junc or not dst_junc or not path_exists(junction_graph, src_junc, dst_junc):
#                     continue
#                 path, cost = run_aco(weighted_graph, src_junc, dst_junc)
#                 if path:
#                     edge_route = convert_junction_path_to_edge_route(path, edge_map)
#                     if edge_route:
#                         writer.writerow([t, vid, ' '.join(edge_route)])
#                         vehicle_el = ET.SubElement(root, "vehicle", {
#                             "id": vid,
#                             "type": "slow_vehicle",
#                             "depart": f"{depart:.2f}"
#                         })
#                         ET.SubElement(vehicle_el, "route", {
#                             "edges": ' '.join(edge_route)
#                         })

#     tree = ET.ElementTree(root)
#     tree.write(output_xml, encoding='utf-8', xml_declaration=True)

# if __name__ == "__main__":
#     main()


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

            for edge in incoming_edges:
                edge_clean = edge.lstrip('-')
                edge_to_junction[edge_clean] = junction

            for edge in outgoing_edges:
                edge_clean = edge.lstrip('-')
                edge_to_junction[edge_clean] = junction

    for j1 in all_junctions:
        for out_edge in outgoing_edges_map.get(j1, []):
            edge_clean = out_edge.lstrip('-')
            for j2, in_edges in incoming_edges_map.items():
                if edge_clean in [e.lstrip('-') for e in in_edges] and j1 != j2:
                    # Valid connection from j1 ➝ j2 via edge
                    junction_graph[j1].add(j2)
                    edge_map[(j1, j2)] = out_edge  # preserve sign

        for in_edge in incoming_edges_map.get(j1, []):
            edge_clean = in_edge.lstrip('-')
            for j0, out_edges in outgoing_edges_map.items():
                if edge_clean in [e.lstrip('-') for e in out_edges] and j0 != j1:
                    # Valid connection from j0 ➝ j1 via edge
                    junction_graph[j0].add(j1)
                    edge_map[(j0, j1)] = in_edge  # preserve sign

    return junction_graph, edge_to_junction, edge_map

def load_distance_matrix(path):
    distance = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        headers = list(map(int, next(reader)[1:]))
        for row in reader:
            from_j = int(row[0])
            distance[from_j] = {}
            for i, val in enumerate(row[1:]):
                try:
                    distance[from_j][headers[i]] = float(val)
                except ValueError:
                    distance[from_j][headers[i]] = float('inf')
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

def run_aco(weighted_graph, source, dest, num_ants=15, num_iterations=40,
            alpha=1.0, beta=2.0, evaporation_rate=0.5, q=100.0):
    pheromone = {u: {v: 1.0 for v in weighted_graph[u]} for u in weighted_graph}
    best_path = None
    best_cost = float('inf')

    for _ in range(num_iterations):
        all_paths = []
        all_costs = []

        for _ in range(num_ants):
            path = [source]
            visited = set(path)
            cost = 0

            while path[-1] != dest:
                current = path[-1]
                if current not in weighted_graph:
                    break

                neighbors = [n for n in weighted_graph[current] if n not in visited]
                if not neighbors:
                    break

                probs = []
                for n in neighbors:
                    tau = pheromone[current][n] ** alpha
                    eta = (1.0 / weighted_graph[current][n]) ** beta
                    probs.append(tau * eta)

                total = sum(probs)
                if total == 0:
                    break

                probs = [p / total for p in probs]
                next_node = random.choices(neighbors, weights=probs)[0]

                path.append(next_node)
                visited.add(next_node)
                cost += weighted_graph[current][next_node]

            if path[-1] == dest:
                all_paths.append(path)
                all_costs.append(cost)
                if cost < best_cost:
                    best_path = path
                    best_cost = cost

        for u in pheromone:
            for v in pheromone[u]:
                pheromone[u][v] *= (1 - evaporation_rate)

        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path) - 1):
                pheromone[path[i]][path[i + 1]] += q / cost

    return best_path, best_cost

def convert_junction_path_to_edge_route(junction_path, edge_map):
    edge_route = []
    for i in range(len(junction_path) - 1):
        j_from = junction_path[i]
        j_to = junction_path[i + 1]
        edge = edge_map.get((j_from, j_to))
        if not edge:
            print(f"⚠️ No edge from {j_from} to {j_to}")
            return None
        edge_route.append(edge)
    return edge_route

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

def extract_vehicle_routes_from_xml(xml_path):
    vehicle_data = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for vehicle in root.findall('vehicle'):
        vid = vehicle.attrib['id']
        depart = float(vehicle.attrib.get('depart', 0))
        route = vehicle.find('route')
        if route is not None:
            edges = route.attrib['edges'].strip().split()
            if len(edges) >= 2:
                source_edge = edges[0].lstrip('-')
                dest_edge = edges[-1].lstrip('-')
                vehicle_data.append((vid, source_edge, dest_edge, depart))
    return vehicle_data

def write_pretty_aco_xml(vehicle_edge_routes, output_path="aco_final_edge_route.xml"):
    root = ET.Element("routes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
    })

    for vehicle_id, edge_list, depart_time in vehicle_edge_routes:
        v_elem = ET.SubElement(root, "vehicle", {
            "id": vehicle_id,
            "type": "slow_vehicle",
            "depart": f"{depart_time:.2f}"
        })
        ET.SubElement(v_elem, "route", {"edges": " ".join(edge_list)})

    rough_string = ET.tostring(root, 'utf-8')
    parsed = minidom.parseString(rough_string)
    pretty_xml = parsed.toprettyxml(indent="  ")

    with open(output_path, "w") as f:
        f.write(pretty_xml)
def run_aco_experiment(
    param_id,
    num_ants, num_iterations, alpha, beta, evaporation_rate, q,
    xml_path, junction_csv, distance_file, density_file
):
    output_csv = f"aco_rerouted_paths_{param_id}.csv"
    output_xml = f"aco_final_edge_route_{param_id}.xml"

    junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
    distances = load_distance_matrix(distance_file)
    vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

    final_edge_routes = []
    last_valid_timestamp = None

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'vehicle_id', 'route'])

        for t in range(0, 14400, 300):
            next_t = t + 300
            try:
                densities = load_density_at_time(density_file, next_t)
            except ValueError:
                continue

            weighted_graph = build_weighted_graph(junction_graph, distances, densities, alpha)

            current_timestamp_routes = []
            for vid, src_edge, dst_edge, depart in vehicle_routes:
                src_junc = edge_to_junction.get(src_edge)
                dst_junc = edge_to_junction.get(dst_edge)
                if not src_junc or not dst_junc or not path_exists(junction_graph, src_junc, dst_junc):
                    continue

                path, cost = run_aco(
                    weighted_graph, src_junc, dst_junc,
                    num_ants=num_ants, num_iterations=num_iterations,
                    alpha=alpha, beta=beta,
                    evaporation_rate=evaporation_rate, q=q
                )
                if path:
                    edge_route = convert_junction_path_to_edge_route(path, edge_map)
                    if edge_route:
                        writer.writerow([t, vid, ' '.join(edge_route)])
                        current_timestamp_routes.append((vid, edge_route, depart))

            if current_timestamp_routes:
                final_edge_routes = current_timestamp_routes
                last_valid_timestamp = t

    if final_edge_routes:
        print(f"\n🟢 Writing final ACO routes from timestamp {last_valid_timestamp}s for param set {param_id}...")
        write_pretty_aco_xml(final_edge_routes, output_xml)
        print(f"✅ Param set {param_id} output:\n - {output_csv}\n - {output_xml}")
    else:
        print(f"⚠️ No valid routes found for param set {param_id}.")

def main():
    xml_path = "../routes/aco_vehicle.xml"
    junction_csv = "../models/junction_io_edges.csv"
    distance_file = "../data/aligned_normalized_distance_matrix.csv"
    density_file = "../simulation_data/runtime_routes.csv"

    # 🧪 Define 10 different parameter combinations
    parameter_sets = [
        {"num_ants": 10, "num_iterations": 30, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.5, "q": 100.0},
        {"num_ants": 15, "num_iterations": 40, "alpha": 1.0, "beta": 2.0, "evaporation_rate": 0.3, "q": 100.0},
        {"num_ants": 20, "num_iterations": 50, "alpha": 1.5, "beta": 2.5, "evaporation_rate": 0.4, "q": 150.0},
        {"num_ants": 10, "num_iterations": 60, "alpha": 1.2, "beta": 1.8, "evaporation_rate": 0.6, "q": 90.0},
        {"num_ants": 12, "num_iterations": 45, "alpha": 1.3, "beta": 2.0, "evaporation_rate": 0.5, "q": 120.0},
        {"num_ants": 18, "num_iterations": 35, "alpha": 1.1, "beta": 2.2, "evaporation_rate": 0.4, "q": 110.0},
        {"num_ants": 14, "num_iterations": 30, "alpha": 1.0, "beta": 2.5, "evaporation_rate": 0.5, "q": 130.0},
        {"num_ants": 16, "num_iterations": 40, "alpha": 0.9, "beta": 1.9, "evaporation_rate": 0.3, "q": 95.0},
        {"num_ants": 11, "num_iterations": 55, "alpha": 1.4, "beta": 2.3, "evaporation_rate": 0.6, "q": 105.0},
        {"num_ants": 13, "num_iterations": 42, "alpha": 1.2, "beta": 2.1, "evaporation_rate": 0.45, "q": 98.0}
    ]

    # 🔁 Run all parameter sets
    for i, params in enumerate(parameter_sets, start=1):
        print(f"\n🔧 Running ACO Experiment {i} with parameters: {params}")
        run_aco_experiment(
            param_id=i,
            xml_path=xml_path,
            junction_csv=junction_csv,
            distance_file=distance_file,
            density_file=density_file,
            **params
        )

if __name__ == "__main__":
    main()



# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_csv = "../models/junction_io_edges.csv"
#     distance_file = "../data/aligned_normalized_distance_matrix.csv"
#     density_file = "../simulation_data/runtime_routes.csv"
#     output_csv = "aco_rerouted_paths.csv"
#     output_xml = "aco_final_edge_route.xml"

#     junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
#     distances = load_distance_matrix(distance_file)
#     vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

#     final_edge_routes = []  # Only for last timestamp
#     last_valid_timestamp = None

#     with open(output_csv, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['timestamp', 'vehicle_id', 'route'])

#         for t in range(0, 14400, 300):
#             next_t = t + 300
#             try:
#                 densities = load_density_at_time(density_file, next_t)
#             except ValueError:
#                 continue

#             weighted_graph = build_weighted_graph(junction_graph, distances, densities)

#             current_timestamp_routes = []
#             for vid, src_edge, dst_edge, depart in vehicle_routes:
#                 src_junc = edge_to_junction.get(src_edge)
#                 dst_junc = edge_to_junction.get(dst_edge)
#                 if not src_junc or not dst_junc or not path_exists(junction_graph, src_junc, dst_junc):
#                     continue

#                 path, cost = run_aco(weighted_graph, src_junc, dst_junc)
#                 if path:
#                     edge_route = convert_junction_path_to_edge_route(path, edge_map)
#                     if edge_route:
#                         writer.writerow([t, vid, ' '.join(edge_route)])
#                         current_timestamp_routes.append((vid, edge_route, depart))

#             if current_timestamp_routes:
#                 final_edge_routes = current_timestamp_routes
#                 last_valid_timestamp = t

#     # ✅ Only write final timestamp routes to XML
#     if final_edge_routes:
#         print(f"\n🟢 Writing final ACO routes from timestamp {last_valid_timestamp}s to XML...")
#         write_pretty_aco_xml(final_edge_routes, output_xml)
#         print(f"✅ Output written to: {output_xml}")
#     else:
#         print("⚠️ No valid ACO routes found for any timestamp.")


# if __name__ == "__main__":
#     main()
