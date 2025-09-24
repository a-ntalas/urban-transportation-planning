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
# --- Prepare first figure: network map ---


# --- Prepare figure with 2 rows (top bigger, bottom smaller) ---
fig = plt.figure(figsize=(6, 8))
gs = GridSpec(2, 1, height_ratios=[2.2, 1], figure=fig)

ax_map = fig.add_subplot(gs[0, 0])     # top: map
ax_table = fig.add_subplot(gs[1, 0])   # bottom: table


# --- Top row: network map ---
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

# ax_map.set_title("Ροή επιβατών ανά γραμμή", fontsize=14)
ax_map.axis('off')

# Colorbar under the map
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_map, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.7)
cbar.set_label("Επιβάτες", fontsize=10)


# --- Bottom row: summary table (μ.ω.τ.) ---
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
for (i, j), cell in tbl_summary.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')


plt.tight_layout()
plt.show()
