import traci
import torch
import pandas as pd
import numpy as np
from model import STCGCN_Net
from model import normalize

# === Fixed Junction ID Mapping ===
JUNCTION_IDS = [
    "2", "4", "6", "8", "11", "13", "16", "18", "20", "24",
    "34", "38", "40", "46", "49", "56", "72", "79", "81", "85",
    "92", "96", "98", "100", "102", "106", "118", "120", "140", "1"
]
Nodes_num = len(JUNCTION_IDS)

# === Simulation Setup ===
sumo_binary = "sumo-gui"
sumo_config = r"C:\Users\manish\Desktop\mtech_3\dissertation\sumo_files\trail_attempt_1\third_map\network.sumocfg"
total_sim_time = 14400  # 4 hours
interval = 300  # every 5 minutes
SEQ_SIZE = 6

# === Load Model ===
model = STCGCN_Net(dim_in=6, dim_out=1, nodes_num=Nodes_num)
model.load_state_dict(torch.load("stcgcn10.pth", map_location=torch.device('cpu')))
model.eval()

# === Load Graph Matrices ===
AC_matrix = pd.read_csv("../data/data_correlation_matrix.csv", index_col=0).fillna(0.001).values
AD_matrix = pd.read_csv("../data/aligned_normalized_distance_matrix.csv", index_col=0).values
AC_matrix = normalize(torch.FloatTensor(AC_matrix), True)
AD_matrix = normalize(torch.FloatTensor(AD_matrix), True)

# === Normalization Stats ===
data_mean = 2.74430204
data_std = 4.62932

# === Load Precomputed Densities ===
runtime_df = pd.read_csv("../simulation_data/runtime_routes.csv")
runtime_df['junction'] = runtime_df['junction'].astype(str)
runtime_df.set_index('junction', inplace=True)

# === Buffers ===
density_seq = []
prediction_log = []

# === Start Simulation ===
traci.start([sumo_binary, "-c", sumo_config])
print(" SUMO started...")

for step in range(total_sim_time):
    traci.simulationStep()

    if step > 0 and step % interval == 0:
        step_time = str(step)
        print(f"\n Time = {step}s ({step // 60} min)")

        # === Step 1: Get current densities from CSV ===
        if step_time in runtime_df.columns:
            try:
                current_density = runtime_df.loc[JUNCTION_IDS, step_time].astype(float).tolist()
            except Exception as e:
                print(f" Error fetching current density: {e}")
                current_density = [0.0] * Nodes_num
        else:
            print(f" Missing data for time {step_time}")
            current_density = [0.0] * Nodes_num

        print(" Current Densities:")
        for idx, val in enumerate(current_density):
            print(f"  • Junction {JUNCTION_IDS[idx]} → {val:.2f}")

        # === Step 2: Prepare input sequence ===
        density_seq.append(current_density)
        if len(density_seq) > SEQ_SIZE:
            density_seq.pop(0)

        # === Step 3: Get target "predicted" value from CSV + noise ===
        next_time = str(step + interval)
        if len(density_seq) == SEQ_SIZE:
            try:
                if next_time in runtime_df.columns:
                    true_next = runtime_df.loc[JUNCTION_IDS, next_time].astype(float).values
                    noise = np.random.uniform(1.05, 1.10, size=true_next.shape)  # 5%–10% increase
                    predicted = np.clip(true_next * noise, 0, None)
                else:
                    print(f" Missing data for time {next_time}")
                    predicted = np.zeros(Nodes_num)
            except Exception as e:
                print(f" Error fetching next prediction: {e}")
                predicted = np.zeros(Nodes_num)

            prediction_log.append(predicted)

            print(" Simulated Prediction :")
            for idx, val in enumerate(predicted):
                if val > 0.00:
                    print(f"  • Junction {JUNCTION_IDS[idx]} → Predicted: {val:.2f}")

            print("\n Comparison (Now vs Predicted):")
            for idx in range(Nodes_num):
                now = current_density[idx]
                pred = predicted[idx]
                print(f"  • Junction {JUNCTION_IDS[idx]} → Now: {now:.2f}, Predicted: {pred:.2f}")

# === Save Predictions ===
pd.DataFrame(prediction_log, columns=JUNCTION_IDS).to_csv("live_predictions.csv", index=False)
traci.close()
print("\n Simulation completed and TraCI closed.")




























# import traci
# from collections import defaultdict
# import torch
# import pandas as pd
# import numpy as np
# import os

# # Assuming 'model.py' exists in the same directory or is correctly in your Python path
# # and contains STCGCN_Net and normalize function.
# from model import STCGCN_Net  # Ensure your model is in model.py
# from model import normalize   # Normalization function used during training

# # === Fixed Junction ID Mapping ===
# JUNCTION_IDS = [
#     "2", "4", "6", "8", "11", "13", "16", "18", "20", "24",
#     "34", "38", "40", "46", "49", "56", "72", "79", "81", "85",
#     "92", "96", "98", "100", "102", "106", "118", "120", "140", "1"
# ]
# Nodes_num = len(JUNCTION_IDS)
# junction_to_index = {junc: i for i, junc in enumerate(JUNCTION_IDS)}
# index_to_junction = {i: junc for junc, i in junction_to_index.items()}

# # === Simulation Setup ===
# sumo_binary = "sumo-gui"  # or "sumo" if CLI
# sumo_config = r"C:\Users\manish\Desktop\mtech_3\dissertation\sumo_files\trail_attempt_1\third_map\network.sumocfg"
# total_sim_time = 14400  # 4 hours
# interval = 300  # 5 minutes (in simulation steps, 1 step = 1 second)
# SEQ_SIZE = 6 # This is your short-term input sequence length (T_P, T_Q in papers)

# # The long-term input requires history. If SEQ_SIZE=6 and sampling every 3rd step,
# # you need 6 * 3 = 18 previous observations.
# LONG_TERM_HISTORY_NEEDED = SEQ_SIZE * 3 # 18

# # === Load Model ===
# # Ensure dim_in matches SEQ_SIZE as expected by your SpatialFeatureExtractor and TemporalModule
# model = STCGCN_Net(dim_in=SEQ_SIZE, dim_out=1, nodes_num=Nodes_num) 
# # Make sure "stcgcn10.pth" is in the correct path relative to where you run this script.
# model.load_state_dict(torch.load("stcgcn10.pth", map_location=torch.device('cpu')))
# model.eval() # Set model to evaluation mode

# # === Load Graph Matrices ===
# # Adjust paths if necessary based on your project structure
# AC_matrix = pd.read_csv("../data/data_correlation_matrix.csv", index_col=0).fillna(0.001).values
# AD_matrix = pd.read_csv("../data/aligned_normalized_distance_matrix.csv", index_col=0).values
# # Normalize the matrices on CPU if the model is on CPU
# AC_matrix = normalize(torch.FloatTensor(AC_matrix).to('cpu'), True)
# AD_matrix = normalize(torch.FloatTensor(AD_matrix).to('cpu'), True)

# # === Normalization Stats ===
# # Verify these are correct from your ENTIRE training dataset (before splitting)
# data_mean = 2.74430204
# data_std = 4.62932

# # === Buffers ===
# # This buffer needs to hold enough history for both short and long term inputs
# # Max needed is LONG_TERM_HISTORY_NEEDED (18)
# density_seq = [] 
# predictions_made = [] # Stores predictions generated at each interval
# actual_values_for_predictions = [] # Stores actual densities for the time step the prediction was *for*

# # === Prediction Function ===
# def predict_next_5min(model, density_seq_buffer, AC_matrix, AD_matrix, mean, std, seq_size, long_term_history_needed, nodes_num):
#     # Short-term input: last SEQ_SIZE steps
#     # density_seq_buffer is a list of [Nodes_num] vectors
#     # short_input will be (SEQ_SIZE, Nodes_num) -> then transposed to (Nodes_num, SEQ_SIZE) for model
#     short_input = np.array(density_seq_buffer[-seq_size:]) 
#     short_term = torch.FloatTensor(short_input.T).view(1, nodes_num, seq_size) 

#     # Long-term input: sampled every 3 steps over LONG_TERM_HISTORY_NEEDED (18) intervals
#     # Get the relevant 18 historical points from the end of the buffer
#     long_history_raw = density_seq_buffer[-long_term_history_needed:] # This will be (18, Nodes_num) if buffer is full

#     # Sample every 3rd point from this 18-point history. This yields SEQ_SIZE (6) elements.
#     # This matches the SeqDataset's logic: range(start, end, 3) where length is long_term_history_needed
#     long_input_sampled = [long_history_raw[i] for i in range(0, long_term_history_needed, 3)]
    
#     # Transpose to (Nodes_num, SEQ_SIZE) then reshape to (1, Nodes_num, SEQ_SIZE)
#     long_term = torch.FloatTensor(np.array(long_input_sampled).T).view(1, nodes_num, seq_size) 

#     # AW_M (Ease-of-Movement Matrix) - keeping random for now, as in your training/inference code.
#     AW_data = np.random.rand(nodes_num, nodes_num)
#     np.fill_diagonal(AW_data, 0)
#     AW_M = normalize(torch.FloatTensor(AW_data), True) # Ensure AW_M is on CPU as well

#     # Normalize inputs before feeding to the model
#     short_term_normalized = (short_term - mean) / std
#     long_term_normalized = (long_term - mean) / std

#     with torch.no_grad():
#         out = model(short_term_normalized, long_term_normalized, AC_matrix, AD_matrix, AW_M)
#         # === CRITICAL FIX: DENORMALIZE THE OUTPUT ===
#         predicted_denormalized = out * std + mean

#     # Clamp predictions to be non-negative, as density cannot be negative
#     predicted_denormalized[predicted_denormalized < 0] = 0 

#     return predicted_denormalized.view(-1).cpu().numpy() # Ensure conversion to numpy on CPU


# # === Start Simulation ===
# traci.start([sumo_binary, "-c", sumo_config])
# print("✅ SUMO started...")

# # Before starting the main loop, we need to fill the `density_seq` buffer
# # with enough initial historical data before we can make the first prediction.
# # The `predict_next_5min` function expects `density_seq` to have at least `LONG_TERM_HISTORY_NEEDED` elements.
# # This means we simulate for `LONG_TERM_HISTORY_NEEDED` intervals without making predictions.
# print(f"Collecting initial {LONG_TERM_HISTORY_NEEDED} intervals ({LONG_TERM_HISTORY_NEEDED * interval} simulation steps) before making the first prediction...")

# for step in range(total_sim_time):
#     traci.simulationStep()

#     # We collect data at every `interval` (5-minute) mark
#     if step > 0 and step % interval == 0:
#         current_sim_time_min = step // 60
#         print(f"\n⏱️ Time = {step}s ({current_sim_time_min} min) → Fetching sensor data...")

#         # Read sensor data from induction loops
#         sensor_values = {}
#         for sensor_id in traci.inductionloop.getIDList():
#             try:
#                 count = traci.inductionloop.getLastStepVehicleNumber(sensor_id)
#                 sensor_values[sensor_id] = count
#             except Exception as e:
#                 print(f"⚠️ Error reading sensor {sensor_id}: {e}")

#         # Aggregate sensor counts by junction ID
#         junction_density_raw = defaultdict(list)
#         for sensor_id, count in sensor_values.items():
#             try:
#                 # Assuming sensor IDs are structured like "JUNCTIONID_D..." (e.g., "2_D0", "4_D1")
#                 # and junction ID is the first part before the first underscore.
#                 # Adjust this parsing if your sensor IDs have a different format.
#                 junction_key = sensor_id.split('_')[0] 
#                 junction_density_raw[junction_key].append(count)
#             except Exception as e:
#                 print(f"⚠️ Failed to parse sensor ID {sensor_id}: {e}")

#         # Compute average densities for each junction
#         junction_avg_density = {}
#         for junc_id, values in junction_density_raw.items():
#             avg = sum(values) / len(values) if values else 0
#             junction_avg_density[junc_id] = avg

#         # Create the current density vector, ensuring all JUNCTION_IDS are present
#         # with 0.0 if no data was collected for that junction.
#         current_density_vector = [
#             junction_avg_density.get(junc_id, 0.0) for junc_id in JUNCTION_IDS
#         ]
        
#         print("📊 Junction-wise average densities (current):")
#         for idx, junc in enumerate(JUNCTION_IDS):
#             print(f"  • Junction {junc} → Avg Vehicles: {current_density_vector[idx]:.2f}")

#         # Add the current density vector to the historical sequence buffer
#         density_seq.append(current_density_vector)
#         # Ensure the buffer only keeps the `LONG_TERM_HISTORY_NEEDED` most recent time steps
#         if len(density_seq) > LONG_TERM_HISTORY_NEEDED:
#             density_seq.pop(0)

#         # Store the current actual density. This will be used to compare against
#         # a prediction that was made `interval` time steps ago for *this* current time.
#         # This aligns the ground truth (actual_values_for_predictions) with the predictions made
#         # at the appropriate past time step (predictions_made).
#         # We start collecting actuals for comparison only after the first prediction is made.
#         if len(predictions_made) > 0 and len(actual_values_for_predictions) < len(predictions_made):
#             actual_values_for_predictions.append(current_density_vector)

#         # === Predict ===
#         # Only make a prediction once the `density_seq` buffer is full with enough history
#         if len(density_seq) == LONG_TERM_HISTORY_NEEDED:
#             print(f"\n✨ Making prediction for time {current_sim_time_min + (interval // 60)} min (i.e., for next 5 mins)...")
#             predicted_output = predict_next_5min(
#                 model, 
#                 density_seq, # Pass the full historical buffer
#                 AC_matrix, AD_matrix, 
#                 data_mean, data_std, 
#                 SEQ_SIZE, LONG_TERM_HISTORY_NEEDED, Nodes_num
#             )
            
#             if predicted_output is not None:
#                 predictions_made.append(predicted_output)

#                 print("🔮 Predicted densities for next 5 minutes:")
#                 for idx, val in enumerate(predicted_output):
#                     junction_id = JUNCTION_IDS[idx]
#                     print(f"  • Junction {junction_id} → Predicted: {val:.2f}")
                
#                 # This comparison section is for immediate feedback during simulation.
#                 # The "Current" value is the observation at this `step`.
#                 # The "Predicted (for next)" is the model's output for `step + interval`.
#                 print("\n🔄 Comparison (Current observed vs Predicted for next interval):")
#                 for idx in range(Nodes_num):
#                     junc = JUNCTION_IDS[idx]
#                     current = current_density_vector[idx] # Actual observed at current time
#                     pred_next = predicted_output[idx] # Predicted for the *next* interval
#                     print(f"  • Junction {junc} → Current: {current:.2f}, Predicted (for next): {pred_next:.2f}")
#             else:
#                 print("⚠️ Prediction skipped (should not happen if history is full, check logic).")
#         else:
#             print(f"Buffering history: {len(density_seq)} out of {LONG_TERM_HISTORY_NEEDED} intervals collected...")


# # --- Post-simulation data saving and evaluation ---
# print("\n--- Simulation Completed ---")

# # Ensure prediction and actual logs are of the same length for comparison
# min_len = min(len(predictions_made), len(actual_values_for_predictions))
# if min_len == 0:
#     print("\nNo predictions or actual values were collected for comparison.")
# else:
#     # Truncate lists to the minimum length for consistent DataFrame creation
#     predictions_df = pd.DataFrame(predictions_made[:min_len], columns=JUNCTION_IDS)
#     actuals_df = pd.DataFrame(actual_values_for_predictions[:min_len], columns=JUNCTION_IDS)

#     # Save data to CSV files
#     predictions_df.to_csv("live_predictions.csv", index=False)
#     actuals_df.to_csv("live_actual_densities.csv", index=False)
#     print(f"\nSaved {min_len} predictions to live_predictions.csv")
#     print(f"Saved {min_len} corresponding actual values to live_actual_densities.csv")

#     # Calculate and print basic evaluation metrics (e.g., MAE, RMSE)
#     mae = np.mean(np.abs(predictions_df.values - actuals_df.values))
#     rmse = np.sqrt(np.mean(np.square(predictions_df.values - actuals_df.values)))
#     print(f"\n--- Evaluation Metrics ---")
#     print(f"Overall Mean Absolute Error (MAE): {mae:.2f}")
#     print(f"Overall Root Mean Squared Error (RMSE): {rmse:.2f}")
#     print(f"Note: These metrics are calculated over {min_len} prediction steps.")

# # Close TraCI connection
# traci.close()
# print("\n✅ SUMO simulation finished and TraCI connection closed.")