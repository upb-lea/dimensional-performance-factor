import os.path

import matplotlib

from utils import infinite_cylinder as ic
from utils import material_processing as materials
from utils import comsol_interface as comsol
from utils.physics import *
from utils.maths import *
from utils.general_functions import load_dict
from matplotlib import pyplot as plt
from meta.plot_settings import colors
from meta import paths
import logging
import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# -------------------------
# Matplotlib settings
# -------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Bitstream Vera Sans",
    'font.size': 11.0,
    'text.latex.preamble': r"\usepackage{upgreek}",
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Bitstream Vera Serif',
    'mathtext.it': 'Bitstream Vera Serif:italic',
    'mathtext.bf': 'Bitstream Vera Serif:bold'
})

# -------------------------
# Problem definition
# -------------------------
result_folder = "performance_factor"


# -------------------------
# Plot setup
# -------------------------
cm = 1 / 2.54
comsol_color = '0.8'
colors = ["tab:pink", "tab:purple", "tab:blue", "tab:cyan", "tab:green", "tab:olive", "tab:red", "tab:orange"]
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9 * cm, 16 * cm), sharex=True)

# Load results
results = load_dict("PF_comparison.json")

f_ = np.array(results["frequencies"])
pv_limit = results["pv_limit"]
R_ = results["radii"]
PF_static = np.array(results["PF_static"])
b_mean_static = np.array(results["b_mean_static"])
PF_dim = np.array(results["PF_dim"])
b_mean_dim = np.array(results["b_mean_dim"])

for i, R in enumerate(R_):
    ax[0].plot(f_ / 1000, PF_static[i], "--", color=colors[i])
    ax[0].plot(f_ / 1000, PF_dim[i], color=colors[i])

    ax[1].plot(f_ / 1000, b_mean_static[i]*1000, "--", color=colors[i])
    ax[1].plot(f_ / 1000, b_mean_dim[i]*1000, label=f"$R$ = {R * 1000} mm", color=colors[i])

line1 = matplotlib.lines.Line2D([0], [0], label=r"$\mathrm{PF_{dim}}$", color='k')
line2 = matplotlib.lines.Line2D([0], [0], label=r"$\mathrm{PF_{static}}$", color='k', dashes=(5, 2))
legend1 = plt.legend(handles=[line1, line2], ncol=1, loc="upper left", bbox_to_anchor=(0, 2.1))
ax[1].legend(ncols=1, loc="upper right", bbox_to_anchor=(1, 1))
plt.gca().add_artist(legend1)

# ax[0].set_ylim((0.5, 12))
ax[0].set_ylabel(r"$\Phi \cdot f$ in V")
ax[1].set_ylabel(r"$\Phi / (\pi R^2)$ in mT")
ax[1].set_xlabel(r"$f$ in kHz")
# ax.set_title(f"Performance Factors at {int(pv_limit/1000)} kW/m³")
ax[0].grid()
ax[1].grid()
plt.tight_layout()
plt.savefig("PF_comparison.pdf")
plt.show()
