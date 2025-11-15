import pandas as pd

# Load your files
DC = pd.read_csv("../data/data_correlation_matrix.csv", index_col=0)
ND = pd.read_csv("../data/raw_distance_matrix.csv", index_col=0)

# Ensure all index and column labels are strings
DC.index = DC.index.astype(str)
DC.columns = DC.columns.astype(str)
ND.index = ND.index.astype(str)
ND.columns = ND.columns.astype(str)

# Use DC's row order as the master ordering
target_order = list(DC.index)

# Reindex normalized distance matrix to match the order of the correlation matrix
ND_aligned = ND.reindex(index=target_order, columns=target_order)

# Sanity checks with debug output
if ND_aligned.shape != DC.shape:
    raise ValueError(f"Shape mismatch: ND_aligned is {ND_aligned.shape}, DC is {DC.shape}")

if list(ND_aligned.index) != list(DC.index):
    raise ValueError("Row index mismatch between aligned ND and DC.")

if list(ND_aligned.columns) != list(DC.columns):
    raise ValueError("Column index mismatch between aligned ND and DC.")

# Save the aligned distance matrix
ND_aligned.to_csv("../data/aligned_raw_distnace_matrix.csv")

print("✅ Distance matrix successfully aligned and saved to 'aligned_raw_distance_matrix.csv'.")
