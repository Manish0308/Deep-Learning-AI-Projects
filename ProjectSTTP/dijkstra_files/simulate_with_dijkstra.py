# import csv
# import xml.etree.ElementTree as ET
# import xml.dom.minidom as minidom
# from collections import defaultdict, deque
# import heapq

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

# def run_dijkstra(weighted_graph, source, dest):
#     pq = [(0, source, [source])]
#     visited = set()

#     while pq:
#         cost, node, path = heapq.heappop(pq)
#         if node == dest:
#             return path, cost
#         if node in visited:
#             continue
#         visited.add(node)
#         for neighbor, edge_cost in weighted_graph.get(node, {}).items():
#             if neighbor not in visited:
#                 heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
#     return None, float('inf')

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

#     # Beautify XML
#     rough_string = ET.tostring(root, 'utf-8')
#     reparsed = minidom.parseString(rough_string)
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(reparsed.toprettyxml(indent="  "))

# def main():
#     xml_path = "../routes/aco_vehicle.xml"
#     junction_csv = "../models/junction_io_edges.csv"
#     distance_file = "../data/aligned_normalized_distance_matrix.csv"
#     density_file = "../simulation_data/runtime_routes.csv"
#     output_csv = "dijkstra_rerouted_paths.csv"
#     output_xml = "dijkstra_final_edge_route.xml"

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

#                 path, cost = run_dijkstra(weighted_graph, source_j, dest_j)
#                 if path:
#                     writer.writerow([t, vehicle_id, ' '.join(path)])
#                     current_result.append((vehicle_id, path, depart))
#             if current_result:
#                 final_routes = current_result

#     write_final_routes(final_routes, output_xml)

# if __name__ == "__main__":
#     main()


import csv
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from collections import defaultdict, deque
import heapq

def parse_edge_list(value):
    return [e.strip() for e in value.split(',') if e.strip()] if value else []

def build_junction_graph(csv_path):
    edge_to_junction = {}
    junction_graph = defaultdict(set)
    incoming_edges_map = {}
    outgoing_edges_map = {}
    edge_map = {}

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            junction = row['junction_id'].strip()
            incoming_edges = parse_edge_list(row.get('incoming_edges', ''))
            outgoing_edges = parse_edge_list(row.get('outgoing_edges', ''))

            incoming_edges_map[junction] = incoming_edges
            outgoing_edges_map[junction] = outgoing_edges

            for edge in incoming_edges + outgoing_edges:
                edge_to_junction[edge.lstrip('-')] = junction

    for j1 in incoming_edges_map:
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
        if time_step not in times:
            raise ValueError("Time step not found in CSV")
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
            if cost < float('inf'):
                weighted[j1][j2] = cost
    return weighted

def run_dijkstra(weighted_graph, source, dest):
    pq = [(0, source, [source])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)
        if node == dest:
            return path, cost
        if node in visited:
            continue
        visited.add(node)
        for neighbor, edge_cost in weighted_graph.get(node, {}).items():
            if neighbor not in visited:
                heapq.heappush(pq, (cost + edge_cost, neighbor, path + [neighbor]))
    return None, float('inf')

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

def convert_junction_path_to_edges(junction_path, edge_map):
    edge_path = []
    for i in range(len(junction_path) - 1):
        edge = edge_map.get((junction_path[i], junction_path[i+1]))
        if edge:
            edge_path.append(edge)
        else:
            return None
    return edge_path

def write_final_routes(final_routes, output_path):
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
    reparsed = minidom.parseString(rough_string)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(reparsed.toprettyxml(indent="  "))

def main():
    xml_path = "../routes/aco_vehicle.xml"
    junction_csv = "../models/junction_io_edges.csv"
    distance_file = "../data/aligned_normalized_distance_matrix.csv"
    density_file = "../simulation_data/runtime_routes.csv"
    output_csv = "dijkstra_rerouted_paths.csv"
    output_xml = "dijkstra_final_edge_route.xml"

    junction_graph, edge_to_junction, edge_map = build_junction_graph(junction_csv)
    distances = load_distance_matrix(distance_file)
    vehicle_routes = extract_vehicle_routes_from_xml(xml_path)

    final_routes = []
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'vehicle_id', 'route'])

        for t in range(0, 14400, 300):
            next_t = t + 300
            try:
                densities = load_density_at_time(density_file, next_t)
            except ValueError:
                break

            weighted_graph = build_weighted_graph(junction_graph, distances, densities)

            current_result = []
            for vehicle_id, source_edge, dest_edge, depart in vehicle_routes:
                source_j = edge_to_junction.get(source_edge)
                dest_j = edge_to_junction.get(dest_edge)

                if not source_j or not dest_j or not path_exists(junction_graph, source_j, dest_j):
                    continue

                junction_path, cost = run_dijkstra(weighted_graph, source_j, dest_j)
                if junction_path:
                    edge_path = convert_junction_path_to_edges(junction_path, edge_map)
                    if edge_path:
                        writer.writerow([t, vehicle_id, ' '.join(edge_path)])
                        current_result.append((vehicle_id, edge_path, depart))

            if current_result:
                final_routes = current_result

    write_final_routes(final_routes, output_xml)
    print(f"✅ Dijkstra routing completed.\n - CSV: {output_csv}\n - XML: {output_xml}")

if __name__ == "__main__":
    main()

