import os.path
from matplotlib import pyplot as plt

from utils import infinite_cylinder as ic
from utils import comsol_interface as comsol
from utils.physics import *
from utils.maths import *
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
result_folder = "linear_material_air_gap_study"
R = 7.5e-3  # radius
n_ag = 1
fs = [500e3, 1000e3]
fs_labels = ["500 kHz", "1 MHz"]
n_sample = 10
FEM_marker = "."

# -------------------------
# Material properties (linear)
# -------------------------
mu_real, mu_imag = 2124, 354
mu = mu_0 * (mu_real - 1j * mu_imag)
eps_real, eps_imag = 79201, 33595
eps = epsilon_0 * (eps_real - 1j * eps_imag)

# -------------------------
# Plot setup
# -------------------------
cm = 1 / 2.54
fig, ax = plt.subplots(4, figsize=(9*cm, 10*cm), sharex=True)
comsol_color = '0.8'

# -------------------------
# Read Comsol flux data
# -------------------------
df_flux_comsol = comsol.read_df_from_comsol_table(
    link2file=os.path.join(paths.comsol_results, f"{result_folder}/flux_{n_ag}.txt"),
    header=["f", "f2", "real_flux", "complex_flux", "abs_complex_flux"]
)

# -------------------------
# Main loop over frequencies
# -------------------------
for i, f in enumerate(fs):
    FEM_label = "FEM" if i == 0 else None

    # -------------------------
    # Magnetic flux density from Comsol
    # -------------------------
    x_B, y_B, B = comsol.read_comsol_2d_circle_field(
        link=os.path.join(paths.comsol_results, f"{result_folder}/fields_{n_ag}/{int(f)}/MagB_horizontal.txt"),
        field_name="MagB",
        R=R
    )
    r_B = np.sqrt(x_B**2 + y_B**2)
    ax[0].plot(1000*r_B[::n_sample], B[::n_sample]*1000, FEM_marker, color=comsol_color, label=FEM_label)

    # -------------------------
    # Total magnetic flux (from Comsol table)
    # -------------------------
    flux_comsol = df_flux_comsol.loc[df_flux_comsol["f"] == f, "abs_complex_flux"].to_numpy()[0]
    print(f"\nmagnetic_flux = {np.round(flux_comsol * 1e6, 3)} µVs (from comsol)")

    # -------------------------
    # Electric field from Comsol
    # -------------------------
    x_E, y_E, E = comsol.read_comsol_2d_circle_field(
        link=os.path.join(paths.comsol_results, f"{result_folder}/fields_{n_ag}/{int(f)}/MagE_horizontal.txt"),
        field_name="MagE",
        R=R
    )
    r_E = np.sqrt(x_E**2 + y_E**2)
    ax[1].plot(1000*r_E[::n_sample], E[::n_sample], FEM_marker, color=comsol_color)

    # -------------------------
    # Magnetic loss density from Comsol
    # -------------------------
    x_p_mag, y_p_mag, p_mag = comsol.read_comsol_2d_circle_field(
        link=os.path.join(paths.comsol_results, f"{result_folder}/fields_{n_ag}/{int(f)}/p_mag_horizontal.txt"),
        field_name="p_mag",
        R=R
    )
    r_p_mag = np.sqrt(x_p_mag**2 + y_p_mag**2)
    ax[2].plot(1000*r_p_mag[::n_sample], p_mag[::n_sample]/1000, FEM_marker, color=comsol_color)
    mean_pv_mag = 4 * integrate_2d(x_p_mag, y_p_mag, np.abs(p_mag)) / (np.pi * R**2)
    print(f"mean mag. loss density: {mean_pv_mag/1000} kW/m³")

    # -------------------------
    # Electric loss density from Comsol
    # -------------------------
    x_p_el, y_p_el, p_el = comsol.read_comsol_2d_circle_field(
        link=os.path.join(paths.comsol_results, f"{result_folder}/fields_{n_ag}/{int(f)}/p_el_horizontal.txt"),
        field_name="p_el",
        R=R
    )
    r_p_el = np.sqrt(x_p_el**2 + y_p_el**2)
    ax[3].plot(1000*r_p_el[::n_sample], p_el[::n_sample]/1000, FEM_marker, color=comsol_color)
    mean_pv_el = 4 * integrate_2d(x_p_el, y_p_el, np.abs(p_el)) / (np.pi * R**2)
    print(f"mean el. loss density: {mean_pv_el/1000} kW/m³")
    print(f"mean loss density: {(mean_pv_mag + mean_pv_el)/1000} kW/m³  (in cross section)")

    # -------------------------
    # IC analytical model
    # -------------------------
    print("\nIC model:\n")
    A = 1
    r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
    flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

    # Rescaling
    A *= flux_comsol / abs(flux_ic)
    r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
    flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

    # -------------------------
    # Post-Processing
    # -------------------------
    pv_mag_ic = f_pv_mag(f, mu.imag, np.abs(H_))
    pv_el_ic = f_pv_el(f, eps.imag, np.abs(E_))
    mean_pv_ic = (integrate_2d_axi_symmetry_field(r_, pv_mag_ic) +
                  integrate_2d_axi_symmetry_field(r_, pv_el_ic)) / (np.pi * R**2)
    print(f"{A = } (IC)")
    print(f"magnetic_flux = {np.round(abs(flux_ic)*1e6, 3)} µVs (from IC)")
    print(f"mean mag. loss density: {integrate_2d_axi_symmetry_field(r_, pv_mag_ic)/(np.pi*R**2)/1000} kW/m³")
    print(f"mean el. loss density: {integrate_2d_axi_symmetry_field(r_, pv_el_ic)/(np.pi*R**2)/1000} kW/m³")
    print(f"mean loss density: {mean_pv_ic/1000} kW/m³")

    # -------------------------
    # Plot IC fields
    # -------------------------
    ax[0].plot(1000*r_, np.abs(mu*H_)*1000, label=fs_labels[i], color=colors[i])
    ax[1].plot(1000*r_, np.abs(E_), color=colors[i])
    ax[2].plot(1000*r_, pv_mag_ic/1000, color=colors[i])
    ax[3].plot(1000*r_, pv_el_ic/1000, color=colors[i])

# -------------------------
# Final plot touches
# -------------------------
ax[0].set_ylabel(r"$B$ / mT")
ax[1].set_ylabel(r"$E$ / $\frac{\mathrm{V}}{\mathrm{m}}$")
ax[2].set_ylabel(r"$p_\mathrm{mag}$ / $\frac{\mathrm{kW}}{\mathrm{m^3}}$")
ax[3].set_ylabel(r"$p_\mathrm{el}$ / $\frac{\mathrm{kW}}{\mathrm{m^3}}$")
ax[3].set_xlabel(r"$r$ / mm")
ax[0].legend(ncols=3, loc='upper center', bbox_to_anchor=[0.4, 1.8])
ax[3].set_xticks([0, 2.5, 5, 7.5])

for a in ax:
    a.grid()

fig.align_labels()
plt.subplots_adjust(wspace=0, hspace=0.1)
plt.tight_layout()

plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_fields_linear.pdf"))
plt.savefig(os.path.join(paths.grafics, "FEM_vs_IC_fields_linear.png"), dpi=600)
plt.show()
