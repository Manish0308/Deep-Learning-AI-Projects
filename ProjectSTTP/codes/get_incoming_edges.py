import xml.etree.ElementTree as ET
import pandas as pd
from collections import defaultdict

# === INPUT FILE ===
NET_FILE = "../xml_files/network_with_30_tls.net.xml"

# === Step 1: Your 30 Junctions ===
JUNCTION_IDS = [
    "2", "4", "6", "8", "11", "13", "16", "18", "20", "24",
    "34", "38", "40", "46", "49", "56", "72", "79", "81", "85",
    "92", "96", "98", "100", "102", "106", "118", "120", "140", "1"
]
JUNCTION_IDS_SET = set(JUNCTION_IDS)

# === Step 2: Parse network and count incoming edges ===
incoming_edge_count = defaultdict(int)

tree = ET.parse(NET_FILE)
root = tree.getroot()

for junction in root.findall("junction"):
    junction_id = junction.get("id")
    
    # Only process target 30 junctions
    if junction_id not in JUNCTION_IDS_SET:
        continue

    incoming_lanes = junction.get("incLanes", "").split()
    
    # Extract unique edges by removing lane suffix (e.g., 'edge1_0' -> 'edge1')
    incoming_edges = {lane.split("_")[0] for lane in incoming_lanes if "_" in lane}
    incoming_edge_count[junction_id] = len(incoming_edges)

# === Step 3: Save to CSV ===
df = pd.DataFrame([
    {"junction": junc, "incoming_edge_count": incoming_edge_count.get(junc, 0)}
    for junc in JUNCTION_IDS
])
df.to_csv("../data/incoming_edges_per_junction.csv", index=False)

print("✅ Saved filtered incoming edge count for 30 junctions to 'data/incoming_edges_per_junction.csv'")
