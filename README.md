# Bus Routes Optimization

## Computing travel-time matrix
1. Load medoid bus stops
    * Reads super_stops_medoids.csv with latitude / longitude.
    * Stores the coordinates in a list.
2. Download Patras road network
    * Uses OSMnx to fetch all drivable roads within 15 km of the center of your stops.
    * The network is a graph: intersections = nodes, roads = edges.
3. Add speed + travel time attributes
    * For each road segment (edge), fill in missing speeds with defaults.
    * Compute travel time = length ÷ speed (in seconds).
4. Snap stops to road network
    * Each medoid (lat/lon) is matched to the nearest road node.
    * This ensures stops are properly connected to the network.
5. Compute pairwise travel times
    * For every stop pair (i, j), run Dijkstra’s algorithm to find the fastest path on the road network.
    * Store the shortest travel time (seconds) in a matrix.
    * If no path exists, mark as inf.
6. Save results
    * as time_matrix.npy (fast binary for Python/OR-Tools).
    * and as time_matrix.csv (easy to inspect in Excel/Pandas).
    