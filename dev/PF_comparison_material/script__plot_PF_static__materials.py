import os.path
import logging
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

from utils.general_functions import load_dict
from meta import paths

# -------------------------
# Logging
# -------------------------
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# -------------------------
# Matplotlib settings
# -------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Bitstream Vera Sans",
    "font.size": 10.0,
    "text.latex.preamble": r"\usepackage{upgreek}",
    "text.latex.preamble": r"\usepackage{siunitx}",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Bitstream Vera Serif",
    "mathtext.it": "Bitstream Vera Serif:italic",
    "mathtext.bf": "Bitstream Vera Serif:bold"
})

# -------------------------
# Colors
# -------------------------
cm = 1 / 2.54
colors = [
    "tab:pink", "tab:purple", "tab:blue", "tab:cyan",
    "tab:green", "tab:olive", "tab:red", "tab:orange"
]
min_y_tick = 12
max_y_tick = 47

# -------------------------
# Load results
# -------------------------
results = load_dict("PF_static_comparison.json")
materials = results["materials"]  # dict of materials

# -------------------------
# Figure setup
# -------------------------
fig, ax = plt.subplots(figsize=(9 * cm, 5 * cm))

# -------------------------
# Plot PF_static curves per material
# -------------------------
for i, (mat_name, mat_data) in enumerate(materials.items()):
    freqs = np.array(mat_data["frequencies"])
    PF_static = np.array(mat_data["PF_static"])/1000

    # Safety check: ensure x and y have the same length
    min_len = min(len(freqs), len(PF_static))
    freqs = freqs[:min_len]
    PF_static = PF_static[:min_len]

    # Plot curve
    ax.plot(freqs / 1000, PF_static, label=mat_name, color=colors[i % len(colors)])

    # Mark optimum
    idx_max = np.argmax(PF_static)
    f_opt = freqs[idx_max] / 1000  # kHz
    PF_opt = PF_static[idx_max]
    ax.plot(f_opt, PF_opt, marker="o", color=colors[i % len(colors)], markersize=6)

    # Draw vertical line down to zero
    ax.vlines(f_opt, 0, PF_opt, color=colors[i % len(colors)], linestyle="--", lw=1)

    # Place label
    if PF_opt > max_y_tick / 2:
        pos_of_label = PF_opt - 3
        vertical_alignment = "top"
    else:
        pos_of_label = PF_opt + 8
        vertical_alignment = "bottom"

    ax.text(
        f_opt, pos_of_label,
        f"{int(round(f_opt))} kHz",
        color=colors[i % len(colors)],
        fontsize=8,
        ha="center",
        va=vertical_alignment,
        rotation=90,
        bbox=dict(facecolor='white', edgecolor='none', pad=1)
    )

# -------------------------
# Axes formatting
# -------------------------
ax.set_ylim((min_y_tick, max_y_tick))
ax.set_xticks([250, 500, 750])
ax.set_xlabel(r"$f$ in kHz")
ax.set_ylabel(r"$\mathcal{P}\!\mathcal{F}$ / \qty{}{\kilo\volt\per\square\meter}")
ax.grid()
ax.legend(loc="upper left", fontsize=8)

# -------------------------
# Save + show
# -------------------------
plt.tight_layout()
plt.savefig(os.path.join("PF_static_comparison_materials.pdf"))
plt.savefig(os.path.join(paths.grafics, "PF_static_comparison_materials.pdf"))
plt.show()
