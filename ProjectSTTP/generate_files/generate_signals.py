import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

NET_FILE = "network_with_30_tls.net.xml"
TLL_FILE = "realworld_traffic_lights.tll.xml"

tree = ET.parse(NET_FILE)
root = tree.getroot()

# Extract all traffic light controlled connections
controlled = {}
for conn in root.findall('connection'):
    tl = conn.get('tl')
    link_index = conn.get('linkIndex')
    if tl:
        if tl not in controlled:
            controlled[tl] = []
        controlled[tl].append(int(link_index))

# Create <additional> root
additional = ET.Element("additional")

# Build traffic light logic
for tl_id, indices in controlled.items():
    num_links = max(indices) + 1
    tlLogic = ET.SubElement(additional, "tlLogic", id=tl_id, type="static", programID="1", offset="0")

    # Alternating phases for even vs odd lanes
    state1 = ''.join(['G' if i % 2 == 0 else 'r' for i in range(num_links)])  # Phase A
    state2 = ''.join(['r' if c == 'G' else 'G' for c in state1])              # Phase B

    ET.SubElement(tlLogic, "phase", duration="3",  state=state1)   # short green
    ET.SubElement(tlLogic, "phase", duration="3",  state='y' * num_links)  # yellow
    ET.SubElement(tlLogic, "phase", duration="40", state=state2)   # long red (opposite green)
    ET.SubElement(tlLogic, "phase", duration="3",  state='y' * num_links)  # yellow

# Write to XML with pretty formatting
xml_str = ET.tostring(additional, encoding="utf-8")
parsed = minidom.parseString(xml_str)
pretty_xml = parsed.toprettyxml(indent="  ")

with open(TLL_FILE, "w", encoding="utf-8") as f:
    f.write(pretty_xml)

print(f"✅ Generated traffic light logic for {len(controlled)} junctions in '{TLL_FILE}'")
