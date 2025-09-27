import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import pickle
from matplotlib.gridspec import GridSpec
from math import isclose

from extractor import Q, A, L, total_passengers, N, V, H_min
from model import I, J, k, n

# --- Load solution ---
with open("solution.pkl", "rb") as f:
    sol = pickle.load(f)
x_sol, y_sol = sol["x"], sol["y"]

# --- Load node coordinates ---
df_coords = pd.read_excel("xls/train/coords.xlsx", index_col=0)
pos = {idx: (row['coordx'], row['coordy']) for idx, row in df_coords.iterrows()}

# --- Aggregate flow per arc ---
flow_on_arc = {a: 0.0 for a in A}
for (r, s) in Q:
    for a in A:
        flow_on_arc[a] += y_sol.get(((r, s), a), 0) or 0

# --- Compute vehicles per line ---
vehicles_per_line = {}
for l, data in L.items():
    total_veh = 0
    breakdown = {}
    line_mode = data['mode']
    for j in J.get(line_mode, []):
        val = int(round(x_sol.get((line_mode, j, l), 0)))
        if val > 0:
            breakdown[f"{line_mode}_{j}"] = val
            total_veh += val
    vehicles_per_line[l] = {"total": total_veh, "breakdown": breakdown}

# --- Prepare figure ---
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(nrows=1, ncols=2, width_ratios=[1.7, 1], figure=fig)
ax_map = fig.add_subplot(gs[0, 0])

# --- Left panel: network map ---
flows = list(flow_on_arc.values()) or [0.0]
min_flow, max_flow = min(flows), max(flows)
if isclose(min_flow, max_flow) and max_flow == 0.0:
    min_flow, max_flow = 0.0, 1.0

norm = plt.Normalize(vmin=min_flow, vmax=max_flow)
cmap = cm.plasma
offset = 0.4  # for parallel arcs

for (o, d), f in flow_on_arc.items():
    if o not in pos or d not in pos:
        continue
    x1, y1 = pos[o]
    x2, y2 = pos[d]
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        continue
    offx, offy = -dy / length * offset, dx / length * offset

    ax_map.plot(
        [x1 + offx, x2 + offx],
        [y1 + offy, y2 + offy],
        color=cmap(norm(f)),
        linewidth=3,
        solid_capstyle='round',
        zorder=1
    )

# Draw nodes
for node, (nx, ny) in pos.items():
    ax_map.scatter(nx, ny, s=450, facecolors='white', edgecolors='black', linewidths=2, zorder=3)
    ax_map.text(nx, ny, str(node), fontsize=9, ha='center', va='center', zorder=4)

ax_map.set_title("Ροή επιβατών ανά γραμμή", fontsize=14)
ax_map.axis('off')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.7)
cbar.set_label("Επιβάτες", fontsize=10)

# --- Right panel: 3 subplots ---
gs_right = GridSpec(3, 1, height_ratios=[1.2, 1, 0.8],
                     left=0.68, right=0.98, top=0.95, bottom=0.05)
ax_vehicle = fig.add_subplot(gs_right[0, 0])
ax_table = fig.add_subplot(gs_right[1, 0])
ax_extra = fig.add_subplot(gs_right[2, 0])

# --- Top: fleet info ---
fleet_rows = [[mode, vtype, n.get((mode, vtype), 0), k[(mode, vtype)]] for mode, vtype in sorted(k.keys())]
df_fleet = pd.DataFrame(fleet_rows, columns=["Μέσο", "Μοντέλο", "Διαθεσιμότητα", "Χωρητικότητα"])

ax_vehicle.axis("off")
tbl_vehicle = ax_vehicle.table(
    cellText=df_fleet.values,
    colLabels=df_fleet.columns,
    colWidths=[0.2, 0.3, 0.25, 0.25],
    cellLoc='center',
    colLoc='center',
    loc='center'
)
tbl_vehicle.auto_set_font_size(False)
tbl_vehicle.set_fontsize(10)
tbl_vehicle.scale(1, 1.4)
ax_vehicle.set_title("Στόλος", fontsize=12)
for (i, j), cell in tbl_vehicle.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')

# --- Middle: assigned vehicles per line ---
rows_summary = []
for i, l in enumerate(sorted(vehicles_per_line.keys())):
    info = vehicles_per_line[l]
    breakdown_str = ", ".join([f"{k.split('_',1)[1]}:{v}" for k, v in info['breakdown'].items()])
    v_value = V[i] if i < len(V) else None
    freq_value = L.get(l, {}).get('frequency_est', 0.0)
    minutes = int(freq_value)
    seconds = round((freq_value - minutes) * 60)
    freq_str = f"{minutes}' {seconds}\""
    rows_summary.append([l, breakdown_str, f"{v_value} km/h", freq_str])

df_summary = pd.DataFrame(rows_summary, columns=["Γραμμή", "Ανά μοντέλο", "μ.ω.τ.", "Συχνότητα"])
ax_table.axis("off")
tbl_summary = ax_table.table(
    cellText=df_summary.values,
    colLabels=df_summary.columns,
    colWidths=[0.15, 0.5, 0.2, 0.2],
    cellLoc='center',
    colLoc='center',
    loc='center'
)
tbl_summary.auto_set_font_size(False)
tbl_summary.set_fontsize(10)
tbl_summary.scale(1, 1.4)
ax_table.set_title("Ανάθεση οχημάτων ανά γραμμή", fontsize=12)
for (i, j), cell in tbl_summary.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')

# --- Bottom: extra info ---
ax_extra.axis("off")
filler_text = (
    f"- Οχήματα σε χρήση: {sum(info['total'] for info in vehicles_per_line.values())}\n"
    f"- Συνολικοί σταθμοί: {len(N)}\n"
    f"- Συνολικοί επιβάτες: {total_passengers}\n"
    f"- Χρονικό παράθυρο: {H_min/60:.2f} ώρες\n"
)
ax_extra.text(
    0.5, 0.5, filler_text, fontsize=10, va='center', ha='center',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round')
)

plt.tight_layout()
plt.show()
