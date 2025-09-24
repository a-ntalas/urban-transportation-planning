import math
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# ---- helpers ----
def project_xy(lat, lon):
    # Simple local projection: scale lon by cos(mean lat) so Euclidean works OK for one city
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    mean_lat_rad = np.deg2rad(np.nanmean(lat))
    x = lon * np.cos(mean_lat_rad)
    y = lat
    return np.column_stack([x, y])

def kmeans_labels(points, k, seed=42):
    if k <= 1:
        return np.zeros(points.shape[0], dtype=int)
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed)
    return km.fit_predict(points)

def split_oversized(points, labels, max_size, seed=42):
    """
    If any cluster exceeds max_size, split it into ceil(size/max_size) subclusters.
    Returns new labels (renumbered 0..C-1).
    """
    new_labels = np.full_like(labels, -1)
    cur = 0
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        size = len(idx)
        if size <= max_size:
            new_labels[idx] = cur
            cur += 1
        else:
            sub_k = math.ceil(size / max_size)
            sub_labels = kmeans_labels(points[idx], sub_k, seed=seed)
            for sub in range(sub_k):
                sub_idx = idx[sub_labels == sub]
                new_labels[sub_idx] = cur
                cur += 1
    return new_labels

def enforce_cap(points, max_size, seed=42):
    """
    1) initial k from cap
    2) iteratively split oversized clusters until all fit the cap
    """
    n = points.shape[0]
    k0 = max(1, math.ceil(n / max_size))
    labels = kmeans_labels(points, k0, seed=seed)

    while True:
        # check sizes
        sizes = pd.Series(labels).value_counts()
        if (sizes <= max_size).all():
            break
        labels = split_oversized(points, labels, max_size, seed=seed)

    # renumber to 0..C-1
    _, inv = np.unique(labels, return_inverse=True)
    return inv

def cluster_medoids(df, labels):
    # pick the row (stop) closest to each cluster centroid = medoid
    pts = df[["latitude","longitude"]].to_numpy()
    medoids = []
    for cid in np.unique(labels):
        idx = np.where(labels == cid)[0]
        cluster_pts = pts[idx]
        centroid = cluster_pts.mean(axis=0, keepdims=True)
        # Euclidean in lat/lon space (OK within a city)
        dists = np.linalg.norm(cluster_pts - centroid, axis=1)
        medoid_local = idx[np.argmin(dists)]
        medoids.append(medoid_local)
    return np.array(medoids, dtype=int)

# ---- usage ----
df = pd.read_csv("xls/stops_clean.csv")  # must have: stop name, latitude, longitude, routes...
points = project_xy(df["latitude"], df["longitude"])

MAX_PER_CLUSTER = 8   # <-- tune this (e.g., 6–10)
labels = enforce_cap(points, max_size=MAX_PER_CLUSTER, seed=42)
df["cluster"] = labels

# choose an actual stop to represent each cluster
medoid_idx = cluster_medoids(df, labels)
cluster_representatives = df.iloc[medoid_idx][["stop_name", "latitude", "longitude", "cluster"]].reset_index(drop=True)

# (Optional) summarize cluster sizes
sizes = df.groupby("cluster").size().rename("size").reset_index()
print(sizes.sort_values("size", ascending=False).head())

# Save results
df.to_csv("xls/stops_with_clusters.csv", index=False)
cluster_representatives.to_csv("xls/super_stops_medoids.csv", index=False)
