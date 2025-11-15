import pandas as pd
import numpy as np
import glob

# === Configuration ===
DENSITY_FOLDER = "simulation_data/"  # Folder where density CSVs are stored
DIST_MATRIX_FILE = "data/normalized_distance_matrix.csv"
OUTPUT_CORR_FILE = "data/data_correlation_matrix.csv"

# === Step 1: Load all junction density files ===
density_files = sorted(glob.glob(f"{DENSITY_FOLDER}/junction_density_*.csv"))
print(f"Found {len(density_files)} files.")

dfs = []
for file in density_files:
    df = pd.read_csv(file)
    df = df.set_index("junction")
    dfs.append(df)

# === Step 2: Stack density values across simulations ===
combined = {}
for junction in dfs[0].index:
    series_list = []
    for df in dfs:
        if junction in df.index:
            series_list.extend(df.loc[junction].values)
        else:
            series_list.extend([0.0] * len(df.columns))
    combined[junction] = series_list

# Create DataFrame (rows = junctions, columns = flattened time series across simulations)
combined_df = pd.DataFrame.from_dict(combined, orient="index")

# === Step 3: Compute correlation matrix and round ===
corr_matrix = combined_df.transpose().corr(method='pearson')
corr_matrix = corr_matrix.round(3)  # Round to 3 decimal points
corr_matrix.to_csv(OUTPUT_CORR_FILE)
print(f"✅ Data correlation matrix saved to {OUTPUT_CORR_FILE}")

# === Optional: Load and show distance matrix ===
try:
    dist_df = pd.read_csv(DIST_MATRIX_FILE, index_col=0)
    print("✅ Distance matrix loaded. You can use it to filter or weight the correlation matrix.")
except FileNotFoundError:
    print("⚠️ Distance matrix file not found. Skipping.")
