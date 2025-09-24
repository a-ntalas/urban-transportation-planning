"""
Line-planning MILP (based on Jánošíková et al., 2012)

Requirements:
    pip install pulp pandas

Run:
    python line_planning_milp.py
"""

from math import isclose
from extractor import N, P, Q, Links, A, L, A_in, A_out, H_min
import pulp
import pandas as pd
import pickle


# ---- User-tunable weights for weighted objective ----
w_time = 0.0   # weight for passenger in-vehicle time objective (3)
w_cost = 0.99    # weight for operator cost objective (4) - proxy via route lengths * vehicles
# w_env  = 0.2    # weight for emissions objective (5) - proxy via emissions per vehicle

# Vehicle classes
I = ['metro', 'train']

# Vehicle types available
J = {
    'metro': ['Gen I', 'Gen II', 'Gen III'],
    'train': ['Standard']
}

# Vehicle capacities (per one-way trip)
k = {
    ('metro', 'Gen I'): 800,     
    ('metro', 'Gen II'): 1000,   
    ('metro', 'Gen III'): 1050,  
    ('train', 'Standard'): 1500 
}

# number of available vehicles of particular mode and type (n_ij)
n = {
    ('metro', 'Gen I'): 30,   
    ('metro', 'Gen II'): 20, 
    ('metro', 'Gen III'): 15,
    ('train', 'Standard'): 10
}  # total fleet available

# ---- Build MILP with PuLP ----
model = pulp.LpProblem("Line_Planning_MILP", pulp.LpMinimize)

# Decision variables
x = {}
for i in I:
    for j in J[i]:
        for l, data in L.items():       # l = line_id, data = dictionary with attributes
            if data['mode'] == i:       # check mode
                x[(i, j, l)] = pulp.LpVariable(f"x_{i}_{j}_{l}", lowBound=0, cat='Integer')

y = {}
for (r,s) in Q:
    for a in A:
        y[((r,s),a)] = pulp.LpVariable(f"y_{r}_{s}_{a}", lowBound=0, cat='Integer')

# Total number of OD pairs
num_od = len(Q)

# Typical (max) link time and route cost
max_link_time = max(Links[a]['t'] for a in A)
max_route_cost = max(L[l]['route_len_m'] for l in L)

# Normalization factors
time_norm = 2 * num_od * max_link_time     # accounts for sum over Q
cost_norm = max_route_cost             # sum over L already reflects multiple contributions

print(time_norm)
print(cost_norm)

# Objectives
obj_time = pulp.lpSum(Links[a]['t'] * y[((r,s),a)] for (r,s) in Q for a in A) / 1

obj_cost = pulp.lpSum(L[l]['route_len_m'] * pulp.lpSum(x[(L[l]['mode'], j, l)] for j in J[L[l]['mode']]) for l in L) / 1

#  obj_env  = pulp.lpSum(emissions_per_vehicle[l] * pulp.lpSum(x[(i,j,l)] for i in I for j in J[i]) for l in L)

model += w_time * obj_time + 300 *  w_cost * obj_cost #+ w_env * obj_env

# Constraints
# ---------------------------
# Capacity constraints (6)  -- mode-aware and safe lookups
# ---------------------------
for a in A:
    line_id = Links[a]['line']
    line_mode = L[line_id]['mode']

    # For capacity we only count x for the mode of the line
    lhs = pulp.lpSum(
        (60 * H_min / L[line_id]['t_turn_sec']) * x.get((i, j, line_id), 0) * k[(i, j)]
        for i in I for j in J[i]
        if i == line_mode  # only include x variables for matching mode
    )

    rhs = pulp.lpSum(y.get(((r, s), a), 0) for (r, s) in Q)
    model += lhs >= rhs, f"capacity_link_{a}"

# ---------------------------
# Demand satisfaction (7) origin outflow
# ---------------------------
for (r, s) in Q:
    model += pulp.lpSum(y.get(((r, s), a), 0) for a in A_out[r]) == P[(r, s)], f"orig_out_{r}_{s}"

# ---------------------------
# Demand satisfaction (8) destination inflow
# ---------------------------
for (r, s) in Q:
    model += pulp.lpSum(y.get(((r, s), a), 0) for a in A_in[s]) == P[(r, s)], f"dest_in_{r}_{s}"

# ---------------------------
# Flow conservation at intermediate nodes (9)
# ---------------------------
for (r, s) in Q:
    for v in N:
        if v != r and v != s:
            model += (
                pulp.lpSum(y.get(((r, s), a), 0) for a in A_in[v])
                == pulp.lpSum(y.get(((r, s), a), 0) for a in A_out[v])
            ), f"flow_cons_{r}_{s}_{v}"

# ---------------------------
# Vehicle availability (10)  -- only sum x over lines that match the mode
# ---------------------------
for i in I:
    for j in J[i]:
        model += (
            pulp.lpSum(x.get((i, j, l), 0) for l, data in L.items() if data['mode'] == i)
            <= n[(i, j)]
        ), f"veh_avail_{i}_{j}"

# ---------------------------
# Frequency upper bound (11) -- mode-aware, use L[l] dict iteration
# ---------------------------
for l, data in L.items():
    line_mode = data['mode']
    t_turn = data['t_turn_sec']

    # sum over vehicle types available for this mode only
    lhs = pulp.lpSum(
        (60 * H_min / t_turn) * x.get((line_mode, j, l), 0)
        for j in J[line_mode]
    )

    model += lhs <= data['max_trips_per_direction_horizon'], f"freq_max_{l}"

# ---------------------------
# For each OD (r,s), forbid arcs entering r and leaving s
# ---------------------------
for (r, s) in Q:
    # no incoming flow to origin r
    model += pulp.lpSum(y.get(((r, s), a), 0) for a in A_in[r]) == 0, f"no_in_to_origin_{r}_{s}"
    # no outgoing flow from destination s
    model += pulp.lpSum(y.get(((r, s), a), 0) for a in A_out[s]) == 0, f"no_out_from_dest_{r}_{s}"

# ---------------------------
# Solve
# ---------------------------
solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=15)
res = model.solve(solver)

print("Solver status:", pulp.LpStatus[model.status])
print("Objective value:", pulp.value(model.objective))

# ---------------------------
# Reporting: assigned vehicles per line (mode-aware)
# ---------------------------
print("\nAssigned vehicles per line:")
rows = []
for l, data in sorted(L.items()):
    total_veh = 0
    veh_breakdown = {}
    line_mode = data['mode']
    for i in I:
        for j in J[i]:
            # only lines that match mode will have variables; use get to be safe
            val = pulp.value(x.get((i, j, l), 0))
            if val is None:
                val = 0
            # sometimes pulp returns float, ensure numeric
            try:
                val_f = float(val)
            except Exception:
                val_f = 0.0

            total_veh += val_f
            if val_f > 0:
                veh_breakdown[f"{i}_{j}"] = int(round(val_f))

    freq = (60 * H_min * total_veh / data['t_turn_sec']) if data['t_turn_sec'] > 0 else 0.0
    freq = H_min/freq

    rows.append({
        'line': l,
        'mode': line_mode,
        'vehicles_assigned': int(round(total_veh)),
        'frequency_est': freq,
        't_turn': data['t_turn_sec'],
        'vehicles_by_type': veh_breakdown
    })

    print(f"  {l} (mode={line_mode}): vehicles={int(round(total_veh))}, frequency={freq:.3f} (trips/horizon), types={veh_breakdown}")

df = pd.DataFrame(rows)

for l, data in L.items():
    total_veh = sum(float(pulp.value(x.get((i, j, l), 0)) or 0) 
                    for i in I for j in J[i])
    
    # frequency in trips/horizon — THIS IS CORRECT
    freq = (60 * H_min * total_veh / data['t_turn_sec']) if data['t_turn_sec'] > 0 else 0.0
    freq = H_min/freq
    # Store it directly in L
    L[l]['frequency_est'] = freq


# ---------------------------
# Link flows (sum over OD pairs) -- safe get
# ---------------------------

print("\nLink flows (sum over OD pairs):")
for a in A:
    total_flow = sum(pulp.value(y.get(((r, s), a), 0)) or 0.0 for (r, s) in Q)
    print(f"  {a}: flow={total_flow:.2f} passengers")

# ---------------------------
# Objective decomposition (if obj_time and obj_cost are defined)
# ---------------------------
print("Objective value =", pulp.value(model.objective))
if 'obj_time' in globals():
    print("obj_time =", pulp.value(obj_time))
if 'obj_cost' in globals():
    print("obj_cost =", pulp.value(obj_cost))
    print("normalized obj_cost:", 300 * pulp.value(obj_cost))

with open("solution.pkl", "wb") as f:
    pickle.dump({
        "x": {k: pulp.value(v) for k, v in x.items()},
        "y": {k: pulp.value(v) for k, v in y.items()}
    }, f)

# Save summary
df.to_csv('line_planning_solution.csv', index=False)
print("\nSaved line assignment summary to line_planning_solution.csv")

__all__ = ['x', 'y', 'I', 'J', 'k', 'n', 'L']