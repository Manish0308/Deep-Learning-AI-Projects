import random
import xml.etree.ElementTree as ET
from xml.dom import minidom

network_file = "network_with_30_tls.net.xml"
trips_file = "trips.xml"

def get_valid_edges(network_file):
    tree = ET.parse(network_file)
    root = tree.getroot()

    edges = []
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        if not edge_id.startswith(":"):  # skip internal junction edges
            edges.append(edge_id)
    return edges

def generate_trips(edges, num_trips=100):
    trips = []
    for i in range(num_trips):
        from_edge, to_edge = random.sample(edges, 2)
        trip = ET.Element("trip")
        trip.set("id", f"vehicle_{i+1}")
        trip.set("type", "slow_vehicle")
        trip.set("depart", str(i * 10))
        trip.set("from", from_edge)
        trip.set("to", to_edge)
        trips.append(trip)
    return trips

def write_trips(trips, filename):
    root = ET.Element("trips")
    for trip in trips:
        root.append(trip)
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(filename, "w") as f:
        f.write(xml_str)
    print(f"✅ Generated {len(trips)} trips in {filename}")

def main():
    edges = get_valid_edges(network_file)
    trips = generate_trips(edges, num_trips=10000)
    write_trips(trips, trips_file)

if __name__ == "__main__":
    main()
