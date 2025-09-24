import pandas as pd
import re

df = pd.read_csv("xls/bus_stops.csv")

# Extract only numbers (bus lines) from description
df["routes"] = df["description"].apply(lambda x: re.findall(r"\b\d+\b", str(x)))

df = df[["stop_name", "routes", "latitude", "longitude"]]
# Save cleaned data
df.to_csv("xls/stops_clean.csv", index=False)

clustered_stops = pd.read_csv("xls/stops_with_clusters.csv")
number_of_clusters = len(pd.unique(clustered_stops['cluster']))

print(number_of_clusters)
