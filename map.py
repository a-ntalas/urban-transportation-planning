import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
from extractor import Q, A, Links

# Load Excel file
df_coords = pd.read_excel("xls/train/coords.xlsx", index_col=0)
pos = {idx: (row['coordx'], row['coordy']) for idx, row in df_coords.iterrows()}

# Assign a color to each line
lines = list({info['line'] for info in Links.values()})
tab10 = cm.get_cmap('tab10')
line_colors = {
    'A': tab10(2),  # green
    'T': tab10(6),  # red
    'B': tab10(3),  # blue
    'C': tab10(0),  # magenta/pink
}
fig, ax = plt.subplots(figsize=(12, 12))

# Draw edges colored by line
offset = 0.5  # offset for parallel lines
for (o, d), info in Links.items():
    line = info['line']
    color = line_colors[line]

    x1, y1 = pos[o]
    x2, y2 = pos[d]

    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length == 0:
        continue
    offx, offy = -dy/length*offset, dx/length*offset

    ax.plot([x1+offx, x2+offx], [y1+offy, y2+offy],
            color=color, linewidth=3)

# Draw nodes
for node, (x, y) in pos.items():
    ax.scatter(x, y, s=450, facecolors='white', edgecolors='black', linewidths=4, zorder=3)
    ax.text(x, y, node, fontsize=8, ha='center', va='center')

# Add legend for lines
for line, color in sorted(line_colors.items()):
    ax.plot([], [], color=color, label=f"{line}", linewidth=3)
ax.legend(
    title="Γραμμές",
    fontsize=12,          # legend label font size
    title_fontsize=16,    # legend title font size
    handlelength=4,       # length of the line symbol
    handleheight=1,       # height of the line symbol
    loc='upper right'     # optional: position of the legend
)

ax.axis('off')
# ax.set_title("Αστικό δίκτυο")
plt.show()
