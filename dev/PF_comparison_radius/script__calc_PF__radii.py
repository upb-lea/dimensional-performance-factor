from utils import infinite_cylinder as ic
from utils import material_processing as materials
from utils.physics import *
from utils.maths import *
from utils.general_functions import save_dict
import logging
import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# -------------------------
# Problem definition
# -------------------------
T_c = 70  # temperature in C
f_ = np.linspace(1e5, 1e6, 30)  # frequency in Hz
R_ = [0.0044, 0.006, 0.0075, 0.01]  # radii in m


pv_limit = 300000  # loss density in W/m^3
PF_tdk = []

PF_dim = []
b_mean_dim = []

PF_static = []
b_mean_static = []

# -------------------------
# Material (nonlinear)
# -------------------------
# init a material database instance
mdb_data = mdb.Data()

complex_permeability = mdb_data.get_complex_permeability(material=mdb.Material.N49,
                                                         data_source=mdb.DataSource.LEA_MTB,
                                                         pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
print(f"\nComplex permeability data: \n {complex_permeability.measurement_data} \n")
complex_permeability.fit_losses()
complex_permeability.fit_permeability_magnitude()
b_common = np.linspace(0, 0.2, 50)

complex_permittivity = mdb_data.get_complex_permittivity(material=mdb.Material.N49,
                                                         data_source=mdb.DataSource.LEA_MTB)
print(f"\nComplex permittivity data: \n {complex_permittivity.measurement_data} \n ")
complex_permittivity.fit_permittivity_magnitude()
complex_permittivity.fit_loss_angle()

# -------------------------
# Performance Factor
# -------------------------
for i, R in enumerate(R_):
    print(f"  radius = {R}, frequencies = {f_}")
    r_ = np.linspace(0, R, 20)  # this is duplicated as a workaround for the static case with the static
    PF_dim_f = []
    b_mean_dim_f = []
    PF_static_f = []
    b_mean_static_f = []
    max_dev = 1e-6
    for f in f_:
        print(f"    {f = }")
        # -------------------------
        # Material fit at temperature and frequency
        # -------------------------
        # Permeability:
        mu_real, mu_imag = complex_permeability.fit_real_and_imaginary_part_at_f_and_T(
            f_op=f,
            T_op=T_c,
            b_vals=np.linspace(0, 0.2, 50)
        )
        # an interpolation in terms of the magnetic field h is needed:
        mu_h = materials.mu_h_from_mu_b((mu_real - j * mu_imag) * mu_0, b_common)

        # Permittivity:
        eps_real, eps_imag = complex_permittivity.fit_real_and_imaginary_part_at_f_and_T(f=f, T=T_c)
        eps = epsilon_0 * (eps_real - complex(0, 1) * eps_imag)

        # -------------------------
        # Static model
        # -------------------------
        flux_static = ic.flux_from_pv__non_linear(pv_limit, R, f, epsilon_0, mu_h)
        PF_static_f.append(2 * np.pi * f * abs(flux_static))
        b_mean_static_f.append(abs(flux_static) / np.pi / R ** 2)

        # -------------------------
        # IC model
        # -------------------------
        flux_ic = ic.flux_from_pv__non_linear(pv_limit, R, f, eps, mu_h)
        PF_dim_f.append(2 * np.pi * f * abs(flux_ic))
        b_mean_dim_f.append(abs(flux_ic) / np.pi / R ** 2)

    # append results for this radius
    PF_static.append(PF_static_f)
    b_mean_static.append(b_mean_static_f)
    PF_dim.append(PF_dim_f)
    b_mean_dim.append(b_mean_dim_f)

# -------------------------
# Store Results
# -------------------------
results = {
    "temperature": T_c,
    "pv_limit": pv_limit,
    "frequencies": list(f_),
    "radii": list(R_),
    "PF_dim": list(PF_dim),
    "b_mean_dim": list(b_mean_dim),
    "PF_static": list(PF_static),
    "b_mean_static": list(b_mean_static)}

save_dict("PF_comparison.json", results)
