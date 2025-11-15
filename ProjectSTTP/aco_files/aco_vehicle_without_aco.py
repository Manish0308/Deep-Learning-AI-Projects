import xml.etree.ElementTree as ET
import traci
import time
import csv
import os
from sumolib import checkBinary

def extract_vehicle_ids_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    vehicles = []
    for v in root.findall("vehicle"):
        vid = v.get("id")
        vehicles.append(vid)
    return vehicles

def get_simulation_end_time(config_path):
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return 14400
    tree = ET.parse(config_path)
    root = tree.getroot()
    for time_node in root.findall("time"):
        for child in time_node:
            if child.tag == "end":
                try:
                    return int(child.attrib["value"])
                except:
                    pass
    return 14400  # fallback

def run_tracking_simulation():
    config_file = "../network.sumocfg"
    vehicle_xml = "../routes/aco_vehicle.xml"
    output_csv = "vehicle_summary.csv"

    vehicle_ids = extract_vehicle_ids_from_xml(vehicle_xml)
    tracked_vehicles = {vid: {"start_time": None, "end_time": None} for vid in vehicle_ids}

    sim_end_time = get_simulation_end_time(config_file)
    sumo_binary = checkBinary("sumo-gui")
    traci.start([sumo_binary, "-c", config_file])
    print("✅ SUMO-GUI started. Waiting for you to press Play...")

    # Wait until simulation starts
    prev_time = traci.simulation.getTime()
    while True:
        traci.simulationStep()
        curr_time = traci.simulation.getTime()
        if curr_time > prev_time:
            print("▶️ Simulation started by user.")
            break

    # Track vehicles
    while traci.simulation.getTime() <= sim_end_time:
        traci.simulationStep()
        sim_time = traci.simulation.getTime()

        for vid in vehicle_ids:
            if vid in traci.vehicle.getIDList():
                if tracked_vehicles[vid]["start_time"] is None:
                    tracked_vehicles[vid]["start_time"] = sim_time
            elif tracked_vehicles[vid]["start_time"] is not None and tracked_vehicles[vid]["end_time"] is None:
                tracked_vehicles[vid]["end_time"] = sim_time

    # Collect results
    summary = []
    for vid in vehicle_ids:
        start_time = tracked_vehicles[vid].get("start_time")
        end_time = tracked_vehicles[vid].get("end_time")

        if start_time is not None and end_time is not None:
            travel_time = end_time - start_time
            summary.append([vid, start_time, end_time, travel_time])

    traci.close()

    # Write results
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VehicleID", "StartTime", "EndTime", "TravelTime"])
        writer.writerows(summary)

    print(f"\n✅ Summary saved to {output_csv}")

# Run it
run_tracking_simulation()
