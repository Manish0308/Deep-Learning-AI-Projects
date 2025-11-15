# aco_routing/preprocess.py
from xml.etree import ElementTree as ET
import pandas as pd

def get_edge_costs_from_density(netfile, density_file):
    tree = ET.parse(netfile)
    root = tree.getroot()

    densities = pd.read_csv(density_file, index_col=0).iloc[:, 0]  # First predicted column

    edge_costs = {}
    for edge in root.findall("edge"):
        if 'function' in edge.attrib and edge.attrib['function'] == 'internal':
            continue
        edge_id = edge.attrib['id']
        for lane in edge.findall('lane'):
            to_node = edge.attrib['to']
            if to_node in densities:
                edge_costs[edge_id] = densities[to_node]
                break
    return edge_costs
