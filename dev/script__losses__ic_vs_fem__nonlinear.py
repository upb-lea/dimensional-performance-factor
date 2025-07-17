import os.path

import matplotlib

from utils import infinite_cylinder as ic
from utils import material_processing as materials
from utils import comsol_interface as comsol
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
result_folder = "non_linear_material_study"

R = 7.5e-3          # radius
T_c = 50            # temperature (unused here)

n_ags = [0, 1, 2, 3]     # number of air gaps
d_ags = [0, 1.8, 1.8, 1.8]  # air gap length in mm

b_mean_goal = 50e-3       # Tesla
flux_goal = b_mean_goal * np.pi * R ** 2
f_min = 0.9e5

# -------------------------
# Material (nonlinear)
# -------------------------
# load material data
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")
print(df_mu[(df_mu["f"] == 100000) & (df_mu["T"] == 50)])
print(df_eps)

# -------------------------
# Plot setup
# -------------------------
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(9*cm, 9*cm))
comsol_color = '0.8'

# -------------------------
# Main loop over air gaps
# -------------------------
for n_ag, d_ag in zip(n_ags, d_ags):
    print(f"{n_ag = }")

    L_c = (29.5 - d_ag) / 1000  # center leg length (m)
    Pvs_IC, Pvs_static = [], []

    # -------------------------
    # Load Comsol flux data
    # -------------------------
    df_flux = comsol.read_df_from_comsol_table(
        link2file=os.path.join(paths.comsol_results, f"{result_folder}/flux_{n_ag}.txt"),
        header=["f", "real_flux", "complex_flux", "abs_complex_flux"]
    )
    df_flux = df_flux[df_flux["f"] > f_min]
    fs = df_flux["f"]

    for f in fs:
        # get reference flux from comsol
        flux_comsol = df_flux.loc[df_flux["f"] == f, "abs_complex_flux"].to_numpy()[0]
        print(f"\n\nmagnetic_flux = {np.round(flux_comsol * 1e6, 3)} µVs (from comsol)")
        print(f"Flux Comsol: {flux_comsol}")

        # Interpolate / Extrapolate material data for f/T-operation point
        eps = materials.eps_from_df(df_eps, f, T_c)
        mu_h = materials.mu_h_from_df(df_mu, f, T_c)

        # -------------------------
        # Static model
        # -------------------------
        B_abs_static = flux_comsol / (np.pi * R ** 2)
        mu_static = materials.mu_complex_T_f_b(df_mu, T=T_c, f=f, b=[B_abs_static])[0]
        H_abs_static = B_abs_static / np.abs(mu_static)
        Pv_static = f_pv_mag(f, mu_static.imag, H_abs_static) * L_c * np.pi * R ** 2

        Pvs_static.append(Pv_static)

        # -------------------------
        # IC model
        # -------------------------
        print("\nIC model:\n")
        # init permeability with static result:
        A = H_abs_static
        r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_static)
        flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)

        print(f"{A = } (IC)")
        print(f"Flux IC: {flux_ic}")

        # iterative rescaling
        while abs((flux_comsol - abs(flux_ic))) / flux_comsol > 1e-6:
            A *= flux_comsol / abs(flux_ic)
            r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_h(np.mean(abs(H_))))
            flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)

            print(f"{A = } (IC)")
            print(f"Flux IC: {abs(flux_ic)}")
            print(f"Flux Comsol: {flux_comsol}")

        # Losses
        pv_mag = f_pv_mag(f, mu_h(abs(H_)).imag, np.abs(H_))
        pv_el = f_pv_el(f, eps.imag, np.abs(E_))
        mean_pv = (integrate_2d_axi_symmetry_field(r_, pv_mag) +
                   integrate_2d_axi_symmetry_field(r_, pv_el)) / (np.pi * R ** 2)
        Pv = mean_pv * L_c * np.pi * R ** 2
        Pvs_IC.append(Pv)

        print(f"mean loss density: {mean_pv / 1000} kW/m³ (IC)")
        print(f"Tot. losses in center leg: {Pv} W (IC)\n")

    # -------------------------
    # FEM losses
    # -------------------------
    df_Pv = comsol.read_df_from_comsol_table(
        link2file=os.path.join(paths.comsol_results, f"{result_folder}/losses_{n_ag}.txt"),
        header=["f", "P_mag", "P_el", "P_v"]
    )
    df_Pv = df_Pv[df_Pv["f"] > f_min]

    # -------------------------
    # Plots
    # -------------------------
    freqs_kHz = np.array(fs) / 1000

    # volume scaling:
    v_c = L_c * np.pi * R**2

    ax.semilogy(freqs_kHz, np.array(Pvs_static) / v_c / 1000, "--", color="black")
    ax.semilogy(freqs_kHz, np.array(Pvs_IC) / v_c / 1000, "-", color="black")
    ax.semilogy(freqs_kHz, df_Pv["P_v"] / v_c / 1000, "x", label=f"{n_ag}", color=colors[n_ag])

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
legend1 = ax.legend(handles=[line1, line2, line3], ncol=1, loc="upper left")

ax.legend(ncols=1, loc="lower right", title="air gaps:")
plt.gca().add_artist(legend1)

ax.grid()
fig.align_labels()
plt.tight_layout()

plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_losses_nonlinear.pdf"))
plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_losses_nonlinear.png"))
plt.show()
