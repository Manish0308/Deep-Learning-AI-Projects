# import traci
# import xml.etree.ElementTree as ET
# import csv
# import time

# def parse_depart_times(route_file):
#     depart_times = {}
#     tree = ET.parse(route_file)
#     root = tree.getroot()

#     for vehicle in root.findall('vehicle'):
#         vid = vehicle.attrib['id']
#         depart = float(vehicle.attrib.get('depart', 0))
#         depart_times[vid] = depart
#     return depart_times

# def simulate_and_record(config_file, route_file, output_csv):
#     depart_times = parse_depart_times(route_file)
#     trip_times = {}

#     # 🚦 Start SUMO-GUI and load scenario
#     print("🚗 Launching SUMO-GUI...")
#     traci.start(["sumo-gui", "-c", config_file])

#     print("🕹️ Waiting for user to press ▶️ Play in SUMO-GUI...")
#     # Wait until simulation time starts advancing
#     while traci.simulation.getTime() == 0:
#         time.sleep(0.5)
#         traci.simulationStep()

#     print("✅ Simulation started. Recording vehicle trip times...")

#     # 🔄 Run simulation
#     while traci.simulation.getMinExpectedNumber() > 0:
#         traci.simulationStep()

#         for vid in traci.simulation.getArrivedIDList():
#             if vid in depart_times:
#                 arrival = traci.simulation.getTime()
#                 trip_times[vid] = {
#                     'depart': depart_times[vid],
#                     'arrival': arrival,
#                     'trip_time': arrival - depart_times[vid]
#                 }

#     traci.close()

#     # 💾 Save results to CSV
#     with open(output_csv, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['vehicle_id', 'depart_time', 'arrival_time', 'trip_time'])
#         for vid, info in sorted(trip_times.items()):
#             writer.writerow([vid, info['depart'], info['arrival'], info['trip_time']])

#     print(f"✅ Final trip times written to: {output_csv}")

# if __name__ == "__main__":
#     config_file = "network.sumocfg"
#     route_file = "aco_final_edge_route.xml"  # or any other final route XML
#     output_csv = "final_trip_times.csv"

#     simulate_and_record(config_file, route_file, output_csv)

import traci
import xml.etree.ElementTree as ET
import csv
import time

def parse_depart_times(route_file):
    depart_times = {}
    tree = ET.parse(route_file)
    root = tree.getroot()

    for vehicle in root.findall('vehicle'):
        vid = vehicle.attrib['id']
        depart = float(vehicle.attrib.get('depart', 0))
        depart_times[vid] = depart
    return depart_times

def numeric_sort_key(item):
    vid = item[0]
    return int(vid.split('_')[1])  # Extracts 10001 from "vehicle_10001"

def simulate_and_record(config_file, route_file, output_csv, max_time=14400):
    depart_times = parse_depart_times(route_file)
    trip_times = {}

    print("🚗 Launching SUMO-GUI...")
    traci.start(["sumo-gui", "-c", config_file])

    print("🕹️ Waiting for user to press ▶️ Play in SUMO-GUI...")
    # Wait for simulation time to advance
    while traci.simulation.getTime() == 0:
        time.sleep(0.5)
        traci.simulationStep()

    print("✅ Simulation started. Recording vehicle trip times...")

    step = 0
    while step < max_time:
        traci.simulationStep()
        step = traci.simulation.getTime()

        # Record vehicles that arrived
        for vid in traci.simulation.getArrivedIDList():
            if vid in depart_times:
                arrival = step
                trip_times[vid] = {
                    'depart': depart_times[vid],
                    'arrival': arrival,
                    'trip_time': arrival - depart_times[vid]
                }

        # Stop if no vehicles are left in the network
        if traci.simulation.getMinExpectedNumber() == 0 and not traci.vehicle.getIDList():
            print("✅ All vehicles have arrived. Stopping simulation.")
            break

    traci.close()

    # Save trip times
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['vehicle_id', 'depart_time', 'arrival_time', 'trip_time'])
        for vid, info in sorted(trip_times.items(), key=numeric_sort_key):
            writer.writerow([vid, info['depart'], info['arrival'], info['trip_time']])

    print(f"✅ Final trip times written to: {output_csv}")

if __name__ == "__main__":
    config_file = "network.sumocfg"
    route_file = "abc_final_edge_route_6.xml"  # change as needed
    output_csv = "final_trip_times_6.csv"

    simulate_and_record(config_file, route_file, output_csv)
