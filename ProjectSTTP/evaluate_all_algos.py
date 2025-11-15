import os
import pandas as pd
import matplotlib.pyplot as plt

# === Path to original baseline trip times ===
original_file = "vehicle_summary.csv"
baseline_df = pd.read_csv(original_file)
baseline_times = baseline_df.set_index("vehicle_id")["trip_time"].to_dict()

# === All folders that contain final_trip_times.csv ===
algorithm_folders = [d for d in os.listdir() if os.path.isdir(d) and os.path.exists(os.path.join(d, "final_trip_times.csv"))]

average_time_savings = []
all_vehicle_savings = []

for folder in sorted(algorithm_folders):
    algo_name = folder.replace("_files", "").upper()  # Clean name
    filepath = os.path.join(folder, "final_trip_times.csv")

    try:
        algo_df = pd.read_csv(filepath)
        algo_df = algo_df[algo_df["vehicle_id"].isin(baseline_times.keys())]

        algo_df["original_time"] = algo_df["vehicle_id"].map(baseline_times)
        algo_df["time_saved"] = algo_df["original_time"] - algo_df["trip_time"]

        avg_saved = algo_df["time_saved"].mean()
        average_time_savings.append((algo_name, avg_saved))

        for _, row in algo_df.iterrows():
            all_vehicle_savings.append((algo_name, row["vehicle_id"], row["time_saved"]))

    except Exception as e:
        print(f"Error reading from {filepath}: {e}")

# === Create DataFrames ===
avg_df = pd.DataFrame(average_time_savings, columns=["Algorithm", "Average Time Saved"])
vehicle_df = pd.DataFrame(all_vehicle_savings, columns=["Algorithm", "Vehicle ID", "Time Saved"])

# === Plot bar graph of average time saved ===
plt.figure(figsize=(10, 6))
plt.bar(avg_df["Algorithm"], avg_df["Average Time Saved"], color='teal')
plt.ylabel("Average Time Saved (seconds)")
plt.title("Average Trip Time Saved by Each Algorithm")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Save to CSV ===
avg_df.to_csv("average_time_savings.csv", index=False)
vehicle_df.to_csv("all_vehicle_time_savings.csv", index=False)
print("✅ Results saved to average_time_savings.csv and all_vehicle_time_savings.csv")
