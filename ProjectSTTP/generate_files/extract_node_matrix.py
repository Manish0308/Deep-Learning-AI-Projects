import xml.etree.ElementTree as ET
import numpy as np
import math
import random
import csv

NET_FILE = "network_with_30_tls.net.xml"
RAW_DIST_MATRIX_FILE = "data/raw_distance_matrix.csv"
NORM_DIST_MATRIX_FILE = "data/normalized_distance_matrix.csv"
CORR_MATRIX_FILE = "data/dummy_correlation_matrix.csv"

def get_nodes(net_file):
    tree = ET.parse(net_file)
    root = tree.getroot()

    nodes = {}
    for node in root.findall("junction"):
        node_id = node.get("id")
        if node_id.startswith(":"):  # skip internal junctions
            continue
        x = float(node.get("x"))
        y = float(node.get("y"))
        nodes[node_id] = (x, y)
    return nodes

def compute_distance_matrix(nodes):
    node_ids = list(nodes.keys())
    num_nodes = len(node_ids)
    dist_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            x1, y1 = nodes[node_ids[i]]
            x2, y2 = nodes[node_ids[j]]
            dist_matrix[i][j] = math.hypot(x2 - x1, y2 - y1)

    return node_ids, dist_matrix

def normalize_matrix(matrix):
    max_val = np.max(matrix)
    if max_val == 0:
        return matrix  # avoid division by zero
    return matrix / max_val

def compute_dummy_correlation_matrix(num_nodes):
    corr_matrix = np.identity(num_nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            corr_value = round(random.uniform(0.1, 1.0), 2)
            corr_matrix[i][j] = corr_matrix[j][i] = corr_value
    return corr_matrix

def write_csv(matrix, node_ids, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([""] + node_ids)
        for node_id, row in zip(node_ids, matrix):
            writer.writerow([node_id] + list(np.round(row, 2)))

# === Main execution ===
nodes = get_nodes(NET_FILE)
node_ids, dist_matrix = compute_distance_matrix(nodes)
norm_dist_matrix = normalize_matrix(dist_matrix)
corr_matrix = compute_dummy_correlation_matrix(len(node_ids))

write_csv(dist_matrix, node_ids, RAW_DIST_MATRIX_FILE)
write_csv(norm_dist_matrix, node_ids, NORM_DIST_MATRIX_FILE)
write_csv(corr_matrix, node_ids, CORR_MATRIX_FILE)

print("✅ Saved: raw, normalized distance and dummy correlation matrices.")
