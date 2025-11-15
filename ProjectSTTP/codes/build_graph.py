# aco_routing/build_graph.py
import networkx as nx
import xml.etree.ElementTree as ET

def build_graph(netfile):
    G = nx.DiGraph()
    tree = ET.parse(netfile)
    root = tree.getroot()
    for edge in root.findall("edge"):
        if 'function' in edge.attrib and edge.attrib['function'] == 'internal':
            continue
        edge_id = edge.attrib['id']
        from_node = edge.attrib['from']
        to_node = edge.attrib['to']
        G.add_edge(from_node, to_node, id=edge_id)
    return G
