import sumolib
import csv

# Path to your network file
net_path = "../xml_files/network_with_30_tls.net.xml"  # Update this if needed
net = sumolib.net.readNet(net_path)

output_file = "junction_io_edges.csv"

with open(output_file, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["junction", "incoming_edges", "outgoing_edges"])

    for node in net.getNodes():
        junction_id = node.getID()
        incoming_edges = [edge.getID() for edge in node.getIncoming()]
        outgoing_edges = [edge.getID() for edge in node.getOutgoing()]

        # Join edge IDs with commas
        incoming_str = ",".join(incoming_edges)
        outgoing_str = ",".join(outgoing_edges)

        writer.writerow([junction_id, incoming_str, outgoing_str])

print(f"✅ Generated {output_file}")
