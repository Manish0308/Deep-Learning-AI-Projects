import xml.etree.ElementTree as ET
import pandas as pd

# === FILES ===
DENSITY_CSV = "junction_density_1.csv"
NET_FILE = "../xml_files/network_with_30_tls.net.xml"
OUTPUT_CSV = "junction_vehicle_count.csv"

# === Step 1: Parse net.xml to get incoming lane count per junction ===
tree = ET.parse(NET_FILE)
root = tree.getroot()

junction_lane_count = {}

for junction in root.findall("junction"):
    junc_id = junction.get("id")
    if junc_id.startswith(":"):
        continue
    inc_lanes = junction.get("incLanes", "").split()
    junction_lane_count[junc_id] = len(inc_lanes)

# === Step 2: Load density CSV and convert to numeric ===
df_density = pd.read_csv(DENSITY_CSV)
df_density.set_index("junction", inplace=True)

# Convert all values to numeric (forcefully)
df_density = df_density.apply(pd.to_numeric, errors='coerce')

# === Step 3: Multiply each row by the number of incoming lanes ===
df_vehicle = df_density.copy()

for junc in df_vehicle.index:
    lane_count = junction_lane_count.get(junc, 1)
    print(f"Junction {junc}: {lane_count} incoming lanes")
    df_vehicle.loc[junc] = df_density.loc[junc] * lane_count

# === Step 4: Save output ===
df_vehicle.reset_index(inplace=True)
df_vehicle.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Converted and saved as: {OUTPUT_CSV}")
