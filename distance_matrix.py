import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx

# === 1. Load medoids ===
medoids = pd.read_csv("xls/super_stops_medoids.csv")  
# Must have columns: stop_name, latitude, longitude

coords = list(zip(medoids["latitude"], medoids["longitude"]))

# === 2. Download Patras road network ===
# Center point = mean of all medoids
center_lat, center_lon = np.mean(medoids["latitude"]), np.mean(medoids["longitude"])

print("Downloading road network for Patras...")
G = ox.graph_from_point((center_lat, center_lon),
                        dist=15000,  # 15 km radius should cover Patras urban area
                        network_type="drive")

# Add edge speeds and travel times (seconds)
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

# === 3. Map each medoid to nearest OSM node ===
print("Mapping medoids to graph nodes...")
nodes = [ox.distance.nearest_nodes(G, lon, lat) for lat, lon in coords]

# === Distance matrix (in meters) ===
n = len(nodes)
dist_matrix = np.zeros((n, n))

print(f"Computing {n}x{n} distance matrix...")
for i in range(n):
    for j in range(n):
        if i != j:
            try:
                dist_matrix[i, j] = nx.shortest_path_length(
                    G, nodes[i], nodes[j], weight="length"
                )
            except nx.NetworkXNoPath:
                dist_matrix[i, j] = np.inf  # unreachable


# === 5. Save results ===
np.save("xls/distance_matrix_medoids.npy", dist_matrix)
pd.DataFrame(dist_matrix).to_csv("xls/distance_matrix_medoids.csv", index=False)

print("Done! Saved distance matrix to 'distance_matrix_medoids.npy' and 'distance_matrix_medoids.csv'.")
