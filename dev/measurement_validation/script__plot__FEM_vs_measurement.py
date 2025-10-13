"""
https://www.tdk-electronics.tdk.com/inf/80/db/fer/pq_40_40.pdf
https://www.tdk-electronics.tdk.com/inf/80/db/fer/pq_26_25.pdf
https://www.tdk-electronics.tdk.com/inf/80/db/fer/pq_20_20.pdf
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
import numpy as np

from utils import comsol_interface as comsol
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
# Colors (consistent per curve type)
# -------------------------
color_fem = "tab:blue"
color_meas = "tab:red"

cm = 1 / 2.54
link2measurements = os.path.join(paths.measurements, "2025_08_18/")

core_names = ["PQ2020", "PQ2625", "PQ4040"]
core_volumes = [2850 / 1e9, 6540 / 1e9, 17580 / 1e9]

# -------------------------
# Create subplots: 2 rows (losses, deviation), N cols (cores)
# -------------------------
fig, axes = plt.subplots(
    2, len(core_names),
    figsize=(18 * cm, 9 * cm),
    sharex=True,
    gridspec_kw={'height_ratios': [1.2, 1]},  # top taller than bottom
)

# Collect all frequencies for x-axis range
all_freqs = []

# FEM
df_Pv = comsol.read_df_from_comsol_table(
    link2file=os.path.join(
        paths.comsol_results,
        f"measurement_validation_2025_10_02/core_losses.txt"
    ),
    header=["f"] + core_names
)
df_Pv_static = comsol.read_df_from_comsol_table(
    link2file=os.path.join(
        paths.comsol_results,
        f"measurement_validation_2025_10_02/core_losses_static.txt"
    ),
    header=["f"] + core_names
)

for col, (core_name, core_volume) in enumerate(zip(core_names, core_volumes)):
    ax_loss = axes[0, col]
    ax_dev = axes[1, col]

    # FEM
    p_fem = df_Pv[core_name] / core_volume / 1000
    f_fem = df_Pv["f"] / 1000
    ax_loss.semilogy(f_fem, p_fem, label="FEM", color=color_fem)
    all_freqs.extend(df_Pv["f"].values)

    # FEM static
    p_fem_static = df_Pv_static[core_name] / core_volume / 1000
    f_fem_static = df_Pv_static["f"] / 1000
    ax_loss.semilogy(f_fem_static, p_fem_static, "--", label="FEM\n(static)", color=color_fem)
    all_freqs.extend(df_Pv_static["f"].values)

    # Measurement
    df = pd.read_csv(link2measurements + f"measurement_{core_name}.csv")
    df = df.sort_values(by='f')
    f_meas = df['f'] / 1000
    p_meas = df['p_density_hyst'] / 1000
    ax_loss.scatter(f_meas, p_meas, s=30, alpha=0.6, label="Meas.", color=color_meas)
    all_freqs.extend(df["f"].values)
    ax_loss.set_ylim(5, 2000)

    # -------------------------
    # Highlight difference in upper plot (losses)
    # -------------------------
    # Interpolate so both curves are evaluated on the same f-grid (f_meas already exists)
    # p_fem_interp_loss = np.interp(f_meas, f_fem, p_fem)
    # p_fem_static_interp_loss = np.interp(f_meas, f_fem_static, p_fem_static)

    ax_loss.fill_between(
        f_fem,
        p_fem,
        p_fem_static,
        color="red",
        alpha=0.2,
        label="Diff."
    )


    # -------------------------
    # Deviation calculation
    # -------------------------
    p_fem_interp = np.interp(f_meas, f_fem, p_fem)
    p_fem_static_interp = np.interp(f_meas, f_fem_static, p_fem_static)

    dev_fem = (p_fem_interp - p_meas) / p_meas * 100
    dev_fem_static = (p_fem_static_interp - p_meas) / p_meas * 100

    ax_dev.plot(f_meas, dev_fem, label="FEM", color=color_fem)
    ax_dev.plot(f_meas, dev_fem_static, "--", label="FEM\n(static)", color=color_fem)
    ax_dev.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax_dev.set_ylim(-70, 20)
    ax_dev.set_yticks([-60, -40, -20, 0, 20])

    # Manufacturer deviation
    # ax_dev.axhline(30, color="black", linestyle="--", linewidth=0.8)
    # ax_dev.axhline(-30, color="black", linestyle="--", linewidth=0.8)

    # -------------------------
    # Highlight the difference
    # -------------------------

    ax_dev.fill_between(
        f_meas,
        dev_fem,
        dev_fem_static,
        color="red",
        alpha=0.2,
        interpolate=True,
        label="Diff."
    )

    # -------------------------
    # Titles
    # -------------------------
    ax_loss.set_title(core_name)

    # -------------------------
    # Mean deviations (in the recommended frequency range)
    # -------------------------
    f_rec = np.linspace(300, 1000, 20)  # in kHz
    rec_p_meas_interp = np.interp(f_rec, f_meas, p_meas)
    rec_p_fem_interp = np.interp(f_rec, f_fem, p_fem)
    rec_p_fem_static_interp = np.interp(f_rec, f_fem_static, p_fem_static)
    rec_dev_fem = (rec_p_fem_interp - rec_p_meas_interp) / rec_p_meas_interp
    rec_dev_fem_static = (rec_p_fem_static_interp - rec_p_meas_interp) / rec_p_meas_interp
    mean_dev_fem = np.mean(rec_dev_fem)
    mean_dev_fem_static = np.mean(rec_dev_fem_static)
    print(f"{core_name}")
    print(f"Average deviation FEM         : {np.round(100*mean_dev_fem, 1)} %")
    print(f"Average deviation FEM (static): {np.round(100*mean_dev_fem_static, 1)} %")

# -------------------------
# Labels, limits, legend
# -------------------------
# Row 1 (losses)
for ax in axes[0, :]:
    ax.set_xlabel("")
axes[0, 0].set_ylabel(r"$\overline{p_\mathrm{v}}$ / $\frac{\mathrm{kW}}{\mathrm{m}^3}$")

# Row 2 (deviation)
for ax in axes[1, :]:
    ax.set_xlabel(r"$f$ / kHz")
axes[1, 0].set_ylabel(r"$100 \cdot \frac{\overline{p_\mathrm{v}^\mathrm{\,FEM}}-\overline{p_\mathrm{v}^\mathrm{\,meas}}}{\overline{p_\mathrm{v}^\mathrm{\,meas}}}$ / $\qty{}{\percent}$")

# Shared x-axis range
fmin = min(all_freqs) / 1000
fmax = max(all_freqs) / 1000
for row in range(2):
    for col in range(len(core_names)):
        axes[row, col].set_xlim(fmin, fmax)
        axes[row, col].grid(True)

# -------------------------
# Titles
# -------------------------
for col, core_name in enumerate(core_names):
    axes[0, col].set_title(core_name)


# -------------------------
# Legends
# -------------------------
# Losses legend next to top row, rightmost plot
axes[0, -1].legend(
    loc="center left",          # position relative to the axes
    bbox_to_anchor=(1.02, 0.5), # slightly to the right
    fontsize=8,
    frameon=True,
    title="Losses"
)

# Deviation legend next to bottom row, rightmost plot
axes[1, -1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize=8,
    frameon=True,
    title="Deviation\nvs. Meas."
)

plt.tight_layout()

# -------------------------
# Save + show
# -------------------------
plt.savefig(os.path.join("losses_FEM_vs_measurement.pdf"))
plt.savefig(os.path.join(paths.grafics, "losses_FEM_vs_measurement.pdf"))
plt.show()

