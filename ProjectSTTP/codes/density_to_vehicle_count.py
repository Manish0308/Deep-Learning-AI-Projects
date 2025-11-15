import pandas as pd

# === Input Files ===
DENSITY_CSV = "../simulation_data/junction_density_1.csv"
EDGE_COUNT_CSV = "../data/incoming_edges_per_junction.csv"
OUTPUT_CSV = "../simulation_data/junction_vehicle_count_1.csv"

# === Load CSVs ===
density_df = pd.read_csv(DENSITY_CSV)
edge_count_df = pd.read_csv(EDGE_COUNT_CSV)

# === Convert edge counts to dict for quick access ===
edge_count_map = dict(zip(edge_count_df["junction"], edge_count_df["incoming_edge_count"]))

# === Function to convert density to vehicle count ===
def convert_density_to_vehicle_count(row):
    junction_id = row["junction"]
    edge_count = edge_count_map.get(junction_id, 0)
    multiplier = edge_count * 3  # 3 lanes per edge

    converted = row.drop("junction") * multiplier
    converted["junction"] = junction_id
    return converted

# === Apply conversion ===
converted_df = density_df.apply(convert_density_to_vehicle_count, axis=1)

# === Reorder columns (junction first) ===
cols = ["junction"] + [col for col in converted_df.columns if col != "junction"]
converted_df = converted_df[cols]

# === Save the result ===
converted_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved estimated vehicle counts to: {OUTPUT_CSV}")
