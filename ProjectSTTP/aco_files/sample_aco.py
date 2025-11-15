# import csv
# import random
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
#         for in_edge in incoming_edges_map.get(j1, []):
#             edge_clean = in_edge.lstrip('-')
#             for j0, out_edges in outgoing_edges_map.items():
#                 if edge_clean in [e.lstrip('-') for e in out_edges] and j0 != j1:
#                     junction_graph[j0].add(j1)

#     for junc in all_junctions:
#         if junc not in junction_graph:
#             junction_graph[junc] = set()

#     return junction_graph, edge_to_junction

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
#         all_paths = []
#         all_costs = []

#         for _ in range(num_ants):
#             path = [source]
#             visited = set(path)
#             cost = 0

#             while path[-1] != dest:
#                 current = path[-1]
#                 if current not in weighted_graph:
#                     break

#                 neighbors = [n for n in weighted_graph[current] if n not in visited]
#                 if not neighbors:
#                     break

#                 probs = []
#                 for n in neighbors:
#                     tau = pheromone[current][n] ** alpha
#                     eta = (1.0 / weighted_graph[current][n]) ** beta
#                     probs.append(tau * eta)

#                 total = sum(probs)
#                 if total == 0:
#                     break

#                 probs = [p / total for p in probs]
#                 next_node = random.choices(neighbors, weights=probs)[0]

#                 path.append(next_node)
#                 visited.add(next_node)
#                 cost += weighted_graph[current][next_node]

#             if path[-1] == dest:
#                 all_paths.append(path)
#                 all_costs.append(cost)
#                 if cost < best_cost:
#                     best_path = path
#                     best_cost = cost

#         for u in pheromone:
#             for v in pheromone[u]:
#                 pheromone[u][v] *= (1 - evaporation_rate)

#         for path, cost in zip(all_paths, all_costs):
#             for i in range(len(path) - 1):
#                 pheromone[path[i]][path[i + 1]] += q / cost

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

# def display_graph(graph):
#     print("\n📊 Junction Graph:")
#     for junc, neighbors in sorted(graph.items()):
#         print(f"{junc} → {sorted(neighbors)}")




# import xml.etree.ElementTree as ET

# def extract_vehicle_routes_from_xml(xml_path):
#     vehicle_data = []
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     for vehicle in root.findall('vehicle'):
#         vid = vehicle.attrib['id']
#         route = vehicle.find('route')
#         if route is not None:
#             edges = route.attrib['edges'].strip().split()
#             if len(edges) >= 2:
#                 source_edge = edges[0].lstrip('-')
#                 dest_edge = edges[-1].lstrip('-')
#                 vehicle_data.append((vid, source_edge, dest_edge))
#     return vehicle_data

# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_graph, edge_to_junction = build_junction_graph("../models/junction_io_edges.csv")
#     display_graph(junction_graph)

#     distances = load_distance_matrix("../data/aligned_normalized_distance_matrix.csv")

#     # Extract all vehicle source-destination pairs from XML
#     vehicle_routes = extract_vehicle_routes_from_xml(xml_path)
#     if not vehicle_routes:
#         print("❌ No valid vehicle routes found in XML.")
#         return

#     # Run simulation from 0 to 14100 (since we use t+300)
#     for t in range(0, 14400, 300):
#         next_t = t + 300
#         print(f"\n⏱️ Simulation time: {t}s ➜ Using density at {next_t}s")

#         try:
#             densities = load_density_at_time("../simulation_data/runtime_routes.csv", next_t)
#         except ValueError:
#             print(f"⚠️ No prediction data available for {next_t}s. Stopping.")
#             break

#         weighted_graph = build_weighted_graph(junction_graph, distances, densities, alpha=1.0)

#         for vehicle_id, source_edge, dest_edge in vehicle_routes:
#             source_junc = edge_to_junction.get(source_edge)
#             dest_junc = edge_to_junction.get(dest_edge)

#             print(f"\n🔁 {vehicle_id} routing: {source_junc} ➜ {dest_junc}")

#             if not source_junc or not dest_junc:
#                 print("❌ Invalid source or destination junction.")
#                 continue
#             if not path_exists(junction_graph, source_junc, dest_junc):
#                 print("❌ No path exists in raw graph.")
#                 continue

#             path, cost = run_aco(weighted_graph, source_junc, dest_junc)

#             if path:
#                 print(f"✅ ACO Path for {vehicle_id} at {t}s: {path}")
#                 print(f"📈 Cost: {cost:.2f}")
#             else:
#                 print(f"❌ ACO failed for {vehicle_id} at {t}s using {next_t}s densities.")


import csv
import random
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

def parse_edge_list(value):
    if not value:
        return []
    return [e.strip() for e in value.split(',') if e.strip()]

def build_junction_graph(csv_path):
    edge_to_junction = {}
    junction_graph = defaultdict(set)
    all_junctions = set()
    incoming_edges_map = {}
    outgoing_edges_map = {}
    edge_map = defaultdict(list)

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
            edge_clean = out_edge.lstrip('-')
            j2 = edge_to_junction.get(edge_clean)
            if j2 and j1 != j2:
                junction_graph[j1].add(j2)
                edge_map[(j1, j2)].append(out_edge)

        for in_edge in incoming_edges_map.get(j1, []):
            edge_clean = in_edge.lstrip('-')
            j0 = edge_to_junction.get(edge_clean)
            if j0 and j0 != j1:
                junction_graph[j0].add(j1)
                edge_map[(j0, j1)].append(in_edge)

    for junc in all_junctions:
        if junc not in junction_graph:
            junction_graph[junc] = set()

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

            if path[-1] == dest and cost < best_cost:
                best_path = path
                best_cost = cost

        for u in pheromone:
            for v in pheromone[u]:
                pheromone[u][v] *= (1 - evaporation_rate)

        if best_path:
            for i in range(len(best_path) - 1):
                pheromone[best_path[i]][best_path[i + 1]] += q / best_cost

    return best_path, best_cost

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
        depart = float(vehicle.attrib.get('depart', 0.0))
        route = vehicle.find('route')
        if route is not None:
            edges = route.attrib['edges'].strip().split()
            if len(edges) >= 2:
                source_edge = edges[0].lstrip('-')
                dest_edge = edges[-1].lstrip('-')
                vehicle_data.append((vid, source_edge, dest_edge, depart))
    return vehicle_data

def convert_junction_path_to_edge_route(junction_path, edge_map):
    edge_route = []
    for i in range(len(junction_path) - 1):
        j1, j2 = junction_path[i], junction_path[i+1]
        edges = edge_map.get((j1, j2), [])
        if not edges:
            print(f"⚠️ No edge from {j1} to {j2}")
            return None
        edge_route.append(edges[0])
    return edge_route

def main():
    xml_path = "../routes/aco_vehicle.xml"
    junction_csv = "../models/junction_io_edges.csv"
    distance_file = "../data/aligned_normalized_distance_matrix.csv"
    density_file = "../simulation_data/runtime_routes.csv"
    output_csv = "aco_rerouted_paths.csv"
    output_xml = "aco_final_edge_route.xml"

    junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
    distances = load_distance_matrix(distance_file)
    vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

    final_results = []  # List of (timestamp, vehicle_id, depart_time, edge_route)

    for t in range(0, 14400, 300):
        next_t = t + 300
        try:
            densities = load_density_at_time(density_file, next_t)
        except ValueError:
            break  # No more prediction data

        weighted_graph = build_weighted_graph(junction_graph, distances, densities)

        current_results = []
        for vid, src_edge, dst_edge, depart in vehicle_routes:
            src_junc = edge_to_junction.get(src_edge)
            dst_junc = edge_to_junction.get(dst_edge)
            if not src_junc or not dst_junc or not path_exists(junction_graph, src_junc, dst_junc):
                continue

            path, cost = run_aco(weighted_graph, src_junc, dst_junc)
            if path:
                edge_route = convert_junction_path_to_edge_route(path, edge_map)
                if edge_route:
                    current_results.append((next_t, vid, depart, edge_route))

        if current_results:
            final_results = current_results  # only keep last timestamp's valid ones

    root = ET.Element("routes", {
        "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/routes_file.xsd"
    })

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'vehicle_id', 'route'])
        for timestamp, vid, depart, edge_route in final_results:
            writer.writerow([timestamp, vid, ' '.join(edge_route)])
            vehicle_el = ET.SubElement(root, "vehicle", {
                "id": vid,
                "type": "slow_vehicle",
                "depart": f"{depart:.2f}"
            })
            ET.SubElement(vehicle_el, "route", {
                "edges": ' '.join(edge_route)
            })

    tree = ET.ElementTree(root)
    tree.write(output_xml, encoding='utf-8', xml_declaration=True)
    print(f"\n✅ Final ACO paths written to:\n- {output_csv}\n- {output_xml}")

if __name__ == "__main__":
    main()

