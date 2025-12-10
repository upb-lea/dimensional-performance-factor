import os.path

import matplotlib

from utils import infinite_cylinder as ic
from hybridmag.utils import comsol_interface as comsol
from utils.physics import *
from utils.maths import *
from matplotlib import pyplot as plt
from meta.plot_settings import colors
from meta import paths


# -------------------------
# Matplotlib settings
# -------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Bitstream Vera Sans",
    'font.size': 10.0,
    'text.latex.preamble': r"\usepackage{upgreek}",
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Bitstream Vera Serif',
    'mathtext.it': 'Bitstream Vera Serif:italic',
    'mathtext.bf': 'Bitstream Vera Serif:bold'
})

# -------------------------
# Problem definition
# -------------------------
result_folder = "old_material_data/linear_material_air_gap_study"

R = 7.5e-3          # radius
T_c = 50            # temperature (unused here)
I_sim = 4           # current in A

n_ags = [0, 1, 2, 3]     # number of air gaps
d_ags = [0, 1.8, 1.8, 1.8]  # air gap length in mm

b_mean_goal = 50e-3       # Tesla
flux_goal = b_mean_goal * np.pi * R ** 2
f_min = 0.9e5

# -------------------------
# Material (linear)
# -------------------------
mu_real, mu_imag = 2124, 354
mu = mu_0 * (mu_real - 1j * mu_imag)

eps_real, eps_imag = 79201, 33595
eps = epsilon_0 * (eps_real - 1j * eps_imag)

# -------------------------
# Plot setup
# -------------------------
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(9*cm, 7*cm))
comsol_color = '0.8'

# -------------------------
# Main loop over air gaps
# -------------------------
for n_ag, d_ag in zip(n_ags, d_ags):
    print(f"{n_ag = }")

    L_c = (29.5 - d_ag) / 1000  # center leg length (m)
    Pvs_IC, Pvs_static, R_IC, R_static, I_same = [], [], [], [], []

    # -------------------------
    # Load Comsol flux data
    # -------------------------
    df_flux = comsol.read_df_from_comsol_table(
        link2file=os.path.join(paths.comsol_results, f"{result_folder}/flux_{n_ag}.txt"),
        header=["f", "f2", "real_flux", "complex_flux", "abs_complex_flux"]
    )
    df_flux = df_flux[df_flux["f"] > f_min]
    fs = df_flux["f"]

    for f in fs:
        flux_comsol = df_flux.loc[df_flux["f"] == f, "abs_complex_flux"].to_numpy()[0]
        print(f"\n\nmagnetic_flux = {np.round(flux_comsol * 1e6, 3)} µVs (from comsol)")

        # -------------------------
        # IC model
        # -------------------------
        print("\nIC model:\n")
        A = 1
        r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
        flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

        # Rescaling
        A *= flux_comsol / abs(flux_ic)
        r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
        flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

        # Losses
        pv_mag = f_pv_mag(f, mu.imag, np.abs(H_))
        pv_el = f_pv_el(f, eps.imag, np.abs(E_))
        mean_pv = (integrate_2d_axi_symmetry_field(r_, pv_mag) +
                   integrate_2d_axi_symmetry_field(r_, pv_el)) / (np.pi * R ** 2)
        Pv = mean_pv * L_c * np.pi * R ** 2
        Pvs_IC.append(Pv)
        R_IC.append(2 * Pv / I_sim ** 2)

        print(f"Flux Comsol: {flux_comsol}")
        print(f"Flux IC: {flux_ic}")
        print(f"{A = } (IC)")
        print(f"mean loss density: {mean_pv / 1000} kW/m³")
        print(f"Tot. losses in center leg: {Pv} W\n")

        # -------------------------
        # Static losses
        # -------------------------
        B_abs_static = flux_comsol / (np.pi * R ** 2)
        H_abs_static = B_abs_static / np.abs(mu)
        Pv_static = f_pv_mag(f, mu.imag, H_abs_static) * L_c * np.pi * R ** 2

        Pvs_static.append(Pv_static)
        R_static.append(2 * Pv_static / I_sim ** 2)
        I_same.append(I_sim * flux_goal / flux_comsol)

    # -------------------------
    # FEM losses
    # -------------------------
    df_Pv = comsol.read_df_from_comsol_table(
        link2file=os.path.join(paths.comsol_results, f"{result_folder}/losses_{n_ag}.txt"),
        header=["f", "f2", "P_mag", "P_el", "P_v"]
    )
    df_Pv = df_Pv[df_Pv["f"] > f_min]
    R_FEM = 2 * df_Pv["P_v"] / I_sim ** 2

    # -------------------------
    # Plots
    # -------------------------
    freqs_kHz = np.array(fs) / 1000
    I_same = np.array(I_same)

    # volume scaling:
    v_c = L_c * np.pi * R**2

    ax.semilogy(freqs_kHz, 0.5 * np.array(R_static) * I_same**2 / v_c / 1000, "--", color="black")
    ax.semilogy(freqs_kHz, 0.5 * np.array(R_IC) * I_same**2 / v_c / 1000, "-", color="black")
    ax.semilogy(freqs_kHz, 0.5 * np.array(R_FEM) * I_same**2 / v_c / 1000, "x", label=f"{n_ag}", color=colors[n_ag])

    print((np.array(Pvs_static) - df_Pv["P_v"].to_numpy()) / df_Pv["P_v"].to_numpy())
    print((np.array(Pvs_IC) - df_Pv["P_v"].to_numpy()) / df_Pv["P_v"].to_numpy())

# -------------------------
# Add shaded region
# -------------------------
f_min_kHz = 300
f_max_kHz = 1000

y_min, y_max = ax.get_ylim()
y_max = 2000
ax.fill_betweenx([y_min, y_max], f_min_kHz, f_max_kHz, color='gray', alpha=0.3)

ax.text(
    f_min_kHz,
    y_min,
    "recommended\nfrequency range\n(from TDK)",
    ha='left', va='bottom',
    fontsize=9, color='black'
)

# -------------------------
# Final plot touches
# -------------------------
ax.set_xlabel(r"$f$ / kHz")
ax.set_ylabel(r"$\overline{p_\mathrm{v}}$ / $\frac{\mathrm{kW}}{\mathrm{m}^3}$")

line1 = matplotlib.lines.Line2D([0], [0], label=r"static", color='k', linestyle="--")
line2 = matplotlib.lines.Line2D([0], [0], label=r"IC", color='k')
line3 = matplotlib.lines.Line2D([0], [0], label=r"FEM", color='k', marker="x", linestyle="")
legend1 = ax.legend(handles=[line1, line2, line3], ncol=1, loc="upper left", fontsize=9)

ax.legend(ncols=1, loc="lower right", title="air gaps:", fontsize=9)
plt.gca().add_artist(legend1)

ax.grid()
fig.align_labels()
plt.tight_layout()

plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_losses_linear.pdf"))
plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_losses_linear.png"))
plt.show()
