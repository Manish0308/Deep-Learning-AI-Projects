import xml.etree.ElementTree as ET
from collections import defaultdict
import pandas as pd

# === INPUT FILES ===
SENSOR_XML_FILE = "xml_files/sensor_output.xml"
NET_FILE = "xml_files/network_with_30_tls.net.xml"
TIME_INTERVAL = 300  # seconds
MAX_SIM_TIME = 14400

# === Step 1: Parse net.xml to build incoming edge to junction mapping ===
edge_to_junction = {}
all_junctions = set()

tree = ET.parse(NET_FILE)
root = tree.getroot()

for junction in root.findall("junction"):
    junction_id = junction.get("id")
    if junction_id.startswith(":"):
        continue  # skip internal SUMO junctions

    all_junctions.add(junction_id)
    incoming_lanes = junction.get("incLanes", "").split()
    for lane_id in incoming_lanes:
        edge_id = lane_id.split("_")[0]  # remove lane index
        edge_to_junction[edge_id] = junction_id

# === Step 2: Parse sensor_output.xml and accumulate data ===
# Mapping: {(junction_id, time): [densities]}
junction_density = defaultdict(list)

sensor_data = ET.parse(SENSOR_XML_FILE)
sensor_root = sensor_data.getroot()

for interval in sensor_root.findall("interval"):
    sensor_id = interval.get("id")
    time = float(interval.get("end"))

    if time % TIME_INTERVAL != 0:
        continue  # only consider data at 300s intervals

    edge_id = sensor_id.split("_")[-1]  # assumes sensor ID includes edge index or number
    for edge_prefix in edge_to_junction:
        if edge_id.startswith(edge_prefix):
            junction_id = edge_to_junction[edge_prefix]
            density = float(interval.get("meanVehicleNumber", "0"))
            junction_density[(junction_id, int(time))].append(density)

# === Step 3: Average and pivot the data ===
records = defaultdict(dict)

for (junction_id, time), densities in junction_density.items():
    avg_density = round(sum(densities) / len(densities), 2)
    records[junction_id][time] = avg_density

# Fill missing times with 0.0 for all valid junctions
all_times = list(range(0, MAX_SIM_TIME + 1, TIME_INTERVAL))
for junction_id in all_junctions:
    for t in all_times:
        records[junction_id].setdefault(t, 0.0)

# Create dataframe
df = pd.DataFrame.from_dict(records, orient="index")
df = df[sorted(df.columns)]  # sort time columns
df.index.name = "junction"
df.reset_index(inplace=True)

# === Save to CSV ===
df.to_csv("simulation_data/runtime_routes.csv", index=False)
print("✅ Saved density data to 'junction_density.csv'")
