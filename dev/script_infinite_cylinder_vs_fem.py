import os.path
from utils import material_processing as materials
from utils import infinite_cylinder as ic
from utils import comsol_interface as comsol
from utils.physics import *
from utils.maths import *
from matplotlib import pyplot as plt
from meta.plot_settings import colors
from utils.maths import integrate_2d
from meta import paths

# Problem definition
R = 7.5e-3
T_c = 50
f = 1000e3
fs = [f]
fig, ax = plt.subplots(6, figsize=(3.5, 6), sharex=True)


# --- Comsol
# --- Magnetic flux density ---
x_B_comsol, y_B_comsol, B_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "MagB_horizontal.txt"), field_name="MagB", R=R)
# Get global quantities by integration
magnetic_flux_comsol = 4 * integrate_2d(x=x_B_comsol, y=y_B_comsol, f=np.abs(B_comsol))
print(f"\nmagnetic_flux = {np.round(magnetic_flux_comsol * 1e6, 3)} µVs (from comsol)")
# 1d plot in dependency of the radius
r_B_comsol = np.sqrt(x_B_comsol ** 2 + y_B_comsol ** 2)
ax[0].plot(1000 * r_B_comsol, B_comsol * 1000, "*", color="tab:grey")

# --- Electric field ---
x_E_comsol, y_E_comsol, E_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "MagE_horizontal.txt"), field_name="MagE", R=R)
# 1d plot in dependency of the radius
r_E_comsol = np.sqrt(x_E_comsol ** 2 + y_E_comsol ** 2)
ax[3].plot(1000 * r_E_comsol, E_comsol, "*", color="tab:grey")

# --- Real permeability ---
x_mu_real_comsol, y_mu_real_comsol, mu_real_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "mu_real_horizontal.txt"),
                                                                                        field_name="MagE", R=R)
# 1d plot in dependency of the radius
r_mu_real_comsol = np.sqrt(x_mu_real_comsol ** 2 + y_mu_real_comsol ** 2)
ax[1].plot(1000 * r_mu_real_comsol, mu_real_comsol, "*", color="tab:grey")

# --- Imag permeability ---
x_mu_imag_comsol, y_mu_imag_comsol, mu_imag_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "mu_imag_horizontal.txt"),
                                                                                        field_name="MagE", R=R)
# 1d plot in dependency of the radius
r_mu_imag_comsol = np.sqrt(x_mu_imag_comsol ** 2 + y_mu_imag_comsol ** 2)
ax[2].plot(1000 * r_mu_imag_comsol, mu_imag_comsol, "*", color="tab:grey")

# --- Magnetic loss density ---
x_p_mag_comsol, y_p_mag_comsol, p_mag_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "p_mag_horizontal.txt"),
                                                                                  field_name="p_mag", R=R)
# 1d plot in dependency of the radius
r_p_mag_comsol = np.sqrt(x_p_mag_comsol ** 2 + y_p_mag_comsol ** 2)
ax[4].plot(1000 * r_p_mag_comsol, p_mag_comsol / 1000, "*", color="tab:grey")

mean_pv_mag_comsol = 4 * integrate_2d(x=x_p_mag_comsol, y=y_p_mag_comsol, f=np.abs(p_mag_comsol)) / np.pi / R ** 2

print(f"mean mag. loss density: {mean_pv_mag_comsol / 1000} kW/m³")

# --- Electric loss density ---
x_p_el_comsol, y_p_el_comsol, p_el_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, "p_el_horizontal.txt"),
                                                                               field_name="p_el", R=R)
# 1d plot in dependency of the radius
r_p_el_comsol = np.sqrt(x_p_el_comsol ** 2 + y_p_el_comsol ** 2)
ax[5].plot(1000 * r_p_el_comsol, p_el_comsol / 1000, "*", color="tab:grey")

mean_pv_el_comsol = 4 * integrate_2d(x=x_p_el_comsol, y=y_p_el_comsol, f=np.abs(p_el_comsol)) / np.pi / R ** 2
print(f"mean el. loss density: {mean_pv_el_comsol / 1000} kW/m³")

print(f"mean loss density: {mean_pv_mag_comsol / 1000 + mean_pv_el_comsol / 1000} kW/m³  (in cross section)")


# --- IC model
# Calculate IC-model for the same
phi_max = magnetic_flux_comsol

# load material data
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")

for i, f in enumerate(fs):
    # Interpolate / Extrapolate material data for f/T-operation point
    eps = materials.eps_from_df(df_eps, f, T_c)
    print(eps / epsilon_0)
    mu_h = materials.mu_h_from_df(df_mu, f, T_c)
    print("\nIC model:\n")

    A = 0
    flux = 0
    while not abs(flux) > phi_max:
        # Calculate the magnetic field for the infinite cylinder
        r_, H_, E_ = ic.r_h_e_(R=R, f=f, A=A, eps=eps, mu_of_h=mu_h)

        # Calcultate the dielectric flux from the material law
        D_ = eps * E_

        # Calculate magnetic flux density from material law
        B_ = mu_h(np.abs(H_)) * H_

        # flux
        flux = integral_2d_axi_symmetry_flux_from_b_(r_, B_)

        pv_mag = f_pv_mag(f, mu_h(np.abs(H_)).imag, np.abs(H_))
        mean_pv_mag = integral_2d_axi_symmetry_flux_from_b_(r_, pv_mag) / np.pi / R ** 2

        pv_el = f_pv_el(f, eps.imag, np.abs(E_))
        mean_pv_el = integral_2d_axi_symmetry_flux_from_b_(r_, pv_el) / np.pi / R ** 2

        A = A + 0.01

    print(f"{A = } (IC)")
    print(f"magnetic_flux = {np.round(abs(flux) * 1e6, 3)} µVs (from analytical IC model)")
    print(f"mean mag. loss density: {mean_pv_mag / 1000} kW/m³")
    print(f"mean el. loss density: {mean_pv_el / 1000} kW/m³")
    print(f"mean loss density: {mean_pv_el / 1000 + mean_pv_mag / 1000} kW/m³")
    print("\n\n")

    # plot IC fields
    ax[0].plot(1000 * r_, np.abs(B_) * 1000, label=f"{int(f / 1000)} kHz", color=colors[i])
    ax[1].plot(1000 * r_, mu_h(np.abs(H_)).real / mu_0, color=colors[i])
    ax[2].plot(1000 * r_, mu_h(np.abs(H_)).imag / mu_0, color=colors[i])
    ax[3].plot(1000 * r_, np.abs(E_), color=colors[i])
    ax[4].plot(1000 * r_, pv_mag / 1000, color=colors[i])
    ax[5].plot(1000 * r_, pv_el / 1000, color=colors[i])

ax[0].set_ylabel(r"$B$ / mT")
ax[1].set_ylabel(r"$\mu_\mathrm{real}$")
ax[2].set_ylabel(r"$\mu_\mathrm{imag}$")
ax[3].set_ylabel(r"$E$ / $\frac{\mathrm{V}}{\mathrm{m}}$")
ax[4].set_ylabel(r"$p_\mathrm{mag}$ / $\frac{\mathrm{kW}}{\mathrm{m³}}$")
ax[5].set_ylabel(r"$p_\mathrm{el}$ / $\frac{\mathrm{kW}}{\mathrm{m³}}$")
ax[5].set_xlabel(r"$r$ / mm")

ax[0].legend()

ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
ax[4].grid()
ax[5].grid()

fig.align_labels()
plt.subplots_adjust(wspace=0, hspace=0.1)
plt.tight_layout()

plt.savefig(os.path.join(paths.grafics, f"FEM_vs_IC_{f}.pdf"))
plt.show()
