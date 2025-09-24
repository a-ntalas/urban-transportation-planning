import pandas as pd

# file path
# xlsx_path = "xls/Qmini.xlsx" 
xlsx_path = "xls/train/Q2.xlsx"
sheet_name = "Sheet1"     

# read the matrix; assume first column and first row are station names
df = pd.read_excel(xlsx_path, sheet_name=sheet_name, index_col=0)
df_numeric = df.apply(pd.to_numeric, errors='coerce').fillna(0)
# Optional: inspect
print(df.shape)
print(df.columns[:40])
print(df.index[:])

# Ensure row index and column names match and are strings
df.index = df.index.astype(str)
df.columns = df.columns.astype(str)


# Build N as the list of station IDs (order follows df.columns)
N = list(df.columns)

# Build P: dictionary of (origin, destination) -> demand
# include only positive demands (skip zeros) to keep the model small
P = {}
for origin in df.index:
    for dest in df.columns:
        val = df.at[origin, dest]
        # Only include non-zero demands (optional)
        if pd.notnull(val) and float(val) > 0:
            P[(str(origin), str(dest))] = float(val)

# Q is the list of OD pairs used in the model
Q = list(P.keys())

print("Number of stations:", len(N))
print("Number of OD pairs with positive demand:", len(Q))
total_passengers = df_numeric.iloc[1:40, 0:40].sum().sum()  # sum over rows, then columns

# Load links
df_links = pd.read_excel("xls/train/links_a_faster.xlsx")
# df_links = pd.read_excel("xls/links_mini.xlsx")

# Access velocities
V = [round(df_links.iloc[0, 8],2),  # I2 -> row 2 (index 1), column I (index 7)
              round(df_links.iloc[9, 8],2), # I11 -> row 11 (index 10), column I
              round(df_links.iloc[20, 8],2), # I22 -> row 22 (index 21), column I
              round(df_links.iloc[31, 8],2)] # I33 -> row 33 (index 32), column I
# Build the link dictionary
Links_forwards = {}
Links_backwards = {}
Links = {}
for idx, row in df_links.iterrows():
    o, d = str(row['o']), str(row['d'])
    a = (o, d)  # directed link
    a_bar = (d, o)

    Links_forwards[a] = {
        'line': str(row['line']),      # line name/ID
        't': float(row['sec']) , # travel time [seconds]
        'len': float(row['m']),         # length [meters]
        'mode': str(row['mode'])
    }

    Links_backwards[a_bar] = {
        'line': str(row['line']),      # line name/ID
        't': float(row['sec']) , # travel time [seconds]
        'len': float(row['m']),         # length [meters]
        'mode': str(row['mode'])
    }
    
    Links[a] = {
        'line': str(row['line']),      # line name/ID
        't': float(row['sec']) , # travel time [seconds]
        'len': float(row['m']) ,        # length [meters]
        'mode': str(row['mode'])
    }

    Links[a_bar] = {
        'line': str(row['line']),      # line name/ID
        't': float(row['sec']) , # travel time [seconds]
        'len': float(row['m']),        # length [meters]
        'mode': str(row['mode'])

    }

# Sets
A = list(Links.keys())                       # list of directed links

# print(Links)
print("Links are", len(A), "\n")

# Initialize dicts of outgoing and incoming arcs per node
A_out = {v: [] for v in N}   # Av+ : arcs leaving v
A_in  = {v: [] for v in N}   # Av- : arcs entering v

for (o, d) in Links.keys():
    A_out[o].append((o,d))
    A_in[d].append((o,d))


from collections import defaultdict

# Parameters
H_min = 240.0          # horizon length in minutes (07:00–10:00)
headway_min = 2        # at most one departure every 5 minutes per direction
# dwell_per_stop_min = 0.3  # dwell time at each stop (example: 18 sec)
layover_sec = 15.0         # layover at terminal (example)

# Group links by line
line_links = defaultdict(list)
for (o, d), data in Links_forwards.items():
    line_id = data['line']
    line_links[line_id].append((o, d))

L = {}
for l, arcs in line_links.items():
    # one-way route length
    route_len_m = sum(Links_forwards[a]['len'] for a in arcs)

    # forward travel time
    forward_time_sec = sum(Links_forwards[a]['t'] for a in arcs)

    # dwell: assume one dwell per link
    n_stops_forward = len(arcs)
    # total_dwell_forward = dwell_per_stop_min * n_stops_forward

    # symmetric return time
    # return_time_min = forward_time_sec
    # total_dwell_return = total_dwell_forward

    # turnaround time
    t_turn_sec = 2*forward_time_sec + layover_sec*2*n_stops_forward

    # max trips per direction in horizon
    max_trips_per_direction = H_min / headway_min

    L[l] = {
        'route_len_m': route_len_m,
        'forward_time_sec': forward_time_sec,
        't_turn_sec': t_turn_sec,
        'max_trips_per_direction_horizon': max_trips_per_direction,
        'stop_count': n_stops_forward,
        'mode': Links_forwards[arcs[0]]['mode']

    }
__all__ = ["N", "P", "Q", "Links", "A", "L", "V", "A_out", "A_in", "H_min", "total_passengers"]
