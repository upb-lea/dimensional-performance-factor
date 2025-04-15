from utils import material_processing as materials
from utils import infinite_cylinder as ic
from utils.physics import *
from matplotlib import pyplot as plt
from meta.plot_settings import colors


# load material data
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")
# print(df_mu)
# print(df_eps)

# Problem definition
R = 7.5e-3
T_c = 50
phi_max = 13e-6
# fs = [500e3, 600e3, 700e3, 800e3, 900e3, 1000e3]
fs = [500e3, 1000e3]

fig, ax = plt.subplots(6, figsize=(4, 8), sharex=True)
for i, f in enumerate(fs):
    # Interpolate / Extrapolate material data for f/T-operation point
    eps = materials.eps_from_df(df_eps, f, T_c)
    mu_h = materials.mu_h_from_df(df_mu, f, T_c)
    print("\nother\n")

    A = 0
    flux = 0
    while not abs(flux) > phi_max:
        # Calculate the magnetic field for the infinite cylinder
        r_, H_, E_ = ic.r_h_e_(R=R, f=f, A=A, eps=eps, mu_of_h=mu_h)

        # Calcultate the dielectric flux from the material law
        D_ = eps * E_

        # Calculate magnetic flux density from material law
        B_ = mu_h(H_) * H_

        # flux
        flux = ic.flux_from_b_(r_, B_)
        print(f"magnetic_flux = {np.round(abs(flux)*1e6, 3)} µVs (from analytical IC model)")

        A = A + 0.1

    print("\nother\n")
    A = 0
    flux_NL = 0
    while not abs(flux_NL) > phi_max:
        # Calculate the magnetic field for the infinite cylinder
        r_NL, H_NL, E_NL = ic.r_h_e_NL(R=R, f=f, A=A, eps=eps, mu_h=mu_h)

        # Calcultate the dielectric flux from the material law
        D_NL = eps * E_NL

        # Calculate magnetic flux density from material law
        B_NL = mu_h(H_NL) * H_NL

        # flux
        flux_NL = ic.flux_from_b_(r_NL, B_NL)
        print(f"magnetic_flux = {np.round(abs(flux_NL)*1e6, 3)} µVs (from analytical IC model)")

        A = A + 0.1

    # plot all fields
    ax[0].plot(r_, np.abs(H_), label=f"{int(f/1000)} kHz", color=colors[i])
    ax[0].plot(r_NL, np.abs(H_NL), "--", label=f"{int(f/1000)} kHz", color=colors[i])
    ax[1].plot(r_, mu_h(np.abs(H_)).real/mu_0, color=colors[i])
    ax[1].plot(r_NL, mu_h(np.abs(H_NL)).real/mu_0, "--", color=colors[i])
    ax[2].plot(r_, mu_h(np.abs(H_)).imag/mu_0, color=colors[i])
    ax[2].plot(r_NL, mu_h(np.abs(H_NL)).imag/mu_0, "--", color=colors[i])
    ax[3].plot(r_, np.abs(B_)*1000, color=colors[i])
    ax[3].plot(r_NL, np.abs(B_NL)*1000, "--", color=colors[i])
    ax[4].plot(r_, np.abs(E_), color=colors[i])
    ax[4].plot(r_NL, np.abs(E_NL), "--", color=colors[i])
    ax[5].plot(r_, np.abs(D_), color=colors[i])
    ax[5].plot(r_NL, np.abs(D_NL), "--", color=colors[i])

ax[0].set_ylabel(r"$H$ in A/m")
ax[1].set_ylabel(r"$\mu_\mathrm{real}$")
ax[2].set_ylabel(r"$\mu_\mathrm{imag}$")
ax[3].set_ylabel(r"$B$ in mT")
ax[4].set_ylabel(r"$E$ in V/m")
ax[5].set_ylabel(r"$D$ in As/m²")

ax[0].legend()

ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
ax[4].grid()
ax[5].grid()

plt.tight_layout()
plt.show()
