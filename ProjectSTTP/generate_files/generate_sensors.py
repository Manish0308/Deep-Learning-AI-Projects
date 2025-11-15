import xml.etree.ElementTree as ET
from xml.dom import minidom

# === INPUT ===
NET_FILE = "network_with_30_tls.net.xml"
OUTPUT_FILE = "edge_sensors.add.xml"

# === Parse the network file ===
tree = ET.parse(NET_FILE)
root = tree.getroot()

# Identify all junctions (nodes that receive edges)
junction_nodes = set()
for edge in root.findall('edge'):
    if edge.get("id").startswith(":") or edge.get("function") == "internal":
        continue
    junction_nodes.add(edge.get("to"))

# Output root
sensors = ET.Element("additional")
sensor_count = 0

for edge in root.findall('edge'):
    edge_id = edge.get('id')

    # Skip internal edges
    if edge_id.startswith(":") or edge.get("function") == "internal":
        continue

    to_node = edge.get("to")
    if to_node not in junction_nodes:
        continue  # Not an incoming edge to a junction

    for lane in edge.findall('lane'):
        lane_id = lane.get('id')
        length = float(lane.get('length'))

        # Create an e2Detector sensor on full lane
        ET.SubElement(sensors, "e2Detector", {
            'id': f'sensor_{sensor_count}',
            'lane': lane_id,
            'pos': '0.0',
            'endPos': str(length),
            'freq': '30',  # Output every 30 seconds
            'file': 'sensor_output.csv',  # Output CSV
            'friendlyPos': 'true',
            'export': 'CSV'
        })
        sensor_count += 1

# Pretty-print and write to file
rough_string = ET.tostring(sensors, encoding="utf-8")
reparsed = minidom.parseString(rough_string)
pretty_xml = reparsed.toprettyxml(indent="  ")

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(pretty_xml)

print(f"✅ Generated {sensor_count} sensors on incoming edges to junctions in '{OUTPUT_FILE}'")
