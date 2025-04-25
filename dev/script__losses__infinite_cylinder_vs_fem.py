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
L_c = (29.5 - 1.8) / 1000  # center leg core length (window height - air gap length)
T_c = 50
# fs = np.array([100e3, 700e3, 1000e3])
fs = np.linspace(100e3, 1000e3, 10)
Pvs_IC = []
Pvs_static = []

# load material data
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")

fig, ax = plt.subplots(2, figsize=(3.5, 4), sharex=True)
comsol_color = '0.8'
for i, f in enumerate(fs):

    # --- Comsol
    # --- Magnetic flux density ---
    x_B_comsol, y_B_comsol, B_comsol = comsol.read_comsol_2d_circle_field(link=os.path.join(paths.comsol_results, f"{int(f)}/MagB_horizontal.txt"), field_name="MagB",
                                                                          R=R)
    # 1d plot in dependency of the radius
    r_B_comsol = np.sqrt(x_B_comsol ** 2 + y_B_comsol ** 2)

    # --- Total magnetic flux by numerical integration ---
    magnetic_flux_comsol = 4 * integrate_2d(x=x_B_comsol, y=y_B_comsol, f=np.abs(B_comsol))
    print(f"\nmagnetic_flux = {np.round(magnetic_flux_comsol * 1e6, 3)} µVs (from comsol)")

    # --- IC model
    # Calculate IC-model for the same
    print("\nIC model:\n")

    # Interpolate / Extrapolate material data for f/T-operation point
    eps = materials.eps_from_df(df_eps, f, T_c)
    mu_h = materials.mu_h_from_df(df_mu, f, T_c)

    A = 0
    flux = 0
    while not abs(flux) > magnetic_flux_comsol:
        # Calculate the magnetic field for the infinite cylinder
        r_, H_, E_ = ic.r_h_e_(R=R, f=f, A=A, eps=eps, mu_of_h=mu_h)

        # Calculate magnetic flux density from material law
        B_ = mu_h(np.abs(H_)) * H_

        # flux
        flux = integral_2d_axi_symmetry_flux_from_b_(r_, B_)

        A = A + 0.01

    # Magnetic loss density
    pv_mag = f_pv_mag(f, mu_h(np.abs(H_)).imag, np.abs(H_))
    mean_pv_mag = integral_2d_axi_symmetry_flux_from_b_(r_, pv_mag) / np.pi / R ** 2

    # Dielectric loss density
    pv_el = f_pv_el(f, eps.imag, np.abs(E_))
    mean_pv_el = integral_2d_axi_symmetry_flux_from_b_(r_, pv_el) / np.pi / R ** 2

    # Total loss density and losses
    mean_pv = mean_pv_el + mean_pv_mag
    Pv = mean_pv * L_c * np.pi * R**2
    Pvs_IC.append(Pv)

    print(f"{A = } (IC)")
    print(f"magnetic_flux = {np.round(abs(flux) * 1e6, 3)} µVs (from analytical IC model)")
    print(f"mean mag. loss density: {mean_pv_mag / 1000} kW/m³")
    print(f"mean el. loss density: {mean_pv_el / 1000} kW/m³")
    print(f"mean loss density: {mean_pv / 1000} kW/m³")
    print(f"Tot. losses in center leg: {Pv} W")
    print("\n\n")


    # --- Static losses
    B_abs_static = magnetic_flux_comsol / np.pi / R ** 2
    print(f"{B_abs_static = }")
    H_abs_static = B_abs_static / np.abs(materials.mu_complex_T_f_b(df_mu=df_mu, T=T_c, f=f, b=[B_abs_static])[0])
    print(f"{H_abs_static = }")
    Pv_static = f_pv_mag(f, mu_h(H_abs_static).imag, H_abs_static) * L_c * np.pi * R**2
    print(f"{Pv_static = }")
    Pvs_static.append(Pv_static)


# Plot losses from 3D simulation:
df_Pv_comsol = comsol.read_df_from_comsol_table(link2file=os.path.join(paths.comsol_results, "core losses.txt"),
                                                header=["lam", "i", "f", "T_c", "f2", "P_mag", "P_el", "P_v"])
ax[0].plot(df_Pv_comsol["f"] / 1000, df_Pv_comsol["P_v"], label=f"3D FEM", color=colors[3])


# Plot losses from IC model
ax[0].plot(np.array(fs) / 1000, Pvs_IC, label=f"IC model", color=colors[0])
ax[1].plot(np.array(fs) / 1000, (np.array(Pvs_IC)-df_Pv_comsol["P_v"])/df_Pv_comsol["P_v"], label=f"IC model", color=colors[0])


# Plot losses from static model
ax[0].plot(np.array(fs) / 1000, Pvs_static,  "--", label=f"Static model", color=colors[0])
ax[1].plot(np.array(fs) / 1000, (np.array(Pvs_static)-df_Pv_comsol["P_v"])/df_Pv_comsol["P_v"],  "--", label=f"Static model", color=colors[0])


ax[0].set_ylabel(r"$P_\mathrm{v}$ / W")
ax[1].set_ylabel(r"rel. dev. from FEM")
ax[1].set_xlabel(r"$f$ / kHz")

ax[0].legend()
# ax[1].legend()

ax[0].grid()
ax[1].grid()

fig.align_labels()
plt.subplots_adjust(wspace=0, hspace=0.1)
plt.tight_layout()

plt.savefig(os.path.join(paths.grafics, f"FEM_vs_IC_losses.pdf"))
plt.savefig(os.path.join(paths.grafics, f"FEM_vs_IC_losses.png"))
plt.show()
