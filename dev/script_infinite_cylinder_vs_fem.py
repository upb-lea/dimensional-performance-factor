from utils import material_processing as materials
from utils import infinite_cylinder as ic
from utils.physics import *
from matplotlib import pyplot as plt


# Problem definition
R = 7.5e-3
T_c = 65
f = 550e3
A = 23.4

# load material data
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")
# print(df_mu)
# print(df_eps)

# Interpolate / Extrapolate material data for f/T-operation point
eps = materials.eps_from_df(df_eps, f, T_c)
# print(eps)

mu_h = materials.mu_h_from_df(df_mu, f, T_c)


# exemplary magnetic (test) field
h_eg = np.linspace(0, 50, 20)
# print(mu_h(h_eg))

fig, ax = plt.subplots(6, sharex=True)
# Calculate the magnetic field for the infinite cylinder
r_, H_, E_ = ic.r_h_e_(R=R, f=f, A=A, eps=eps, mu_h=mu_h)
ax[0].plot(r_, H_)

# Plot permeability distribution
ax[1].plot(r_, mu_h(H_).real/mu_0)
ax[2].plot(r_, mu_h(H_).imag/mu_0)

# Calculate magnetic flux density from material law
B_ = mu_h(H_) * H_
ax[3].plot(r_, B_*1000)

# flux
flux = ic.flux_from_b_(r_, B_)
print(f"magnetic_flux = {np.round(abs(flux)*1e6, 3)} µVs (from analytical IC model)")

# Plot the electric field
ax[4].plot(r_, E_)

# Calcultate the dielectric flux from the material law
D_ = eps * E_
ax[5].plot(r_, D_)


ax[0].set_ylabel(r"$H$ in A/m")
ax[1].set_ylabel(r"$\mu_\mathrm{real}$")
ax[2].set_ylabel(r"$\mu_\mathrm{imag}$")
ax[3].set_ylabel(r"$B$ in mT")
ax[4].set_ylabel(r"$E$ in V/m")
ax[5].set_ylabel(r"$D$ in As/m²")

ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
ax[4].grid()
ax[5].grid()

plt.show()
