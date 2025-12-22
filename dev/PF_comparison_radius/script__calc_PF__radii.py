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
result_folder = "performance_factor"
T_c = 100  # temperature
# T_c = 50  # temperature

# f_ = np.array([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6])
f_ = np.linspace(1e5, 1e6, 30)
# f_ = np.array([1e5, 3e5, 5e5, 7e5, 9e5])
R_ = [0.004, 0.006, 0.0075, 0.01]
# R_ = [0.004, 0.0075]

pv_limit = 300000
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
                                                         data_source=mdb.DataSource.TDK_MDT,
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
    print(f"\n{R = }\n")
    r_ = np.linspace(0, R, 20)  # this is duplicated as a workaround for the static case with the static
    PF_dim_f = []
    b_mean_dim_f = []
    PF_static_f = []
    b_mean_static_f = []
    max_dev = 1e-6
    for f in f_:
        print(f"\n{f = }")
        # -------------------------
        # Material fit at temperature and frequency
        # -------------------------
        # Permeability:
        mu_real, mu_imag = complex_permeability.fit_real_and_imaginary_part_at_f_and_T(f_op=f,
                                                                                       T_op=T_c,
                                                                                       b_vals=np.linspace(0, 0.2, 50))
        # an interpolation in terms of the magnetic field h is needed:
        mu_h = materials.mu_h_from_mu_b((mu_real - j * mu_imag) * mu_0, b_common)

        # Permittivity:
        eps_real, eps_imag = complex_permittivity.fit_real_and_imaginary_part_at_f_and_T(f=f, T=T_c)
        eps = epsilon_0 * (eps_real - complex(0, 1) * eps_imag)

        # -------------------------
        # Static model
        # -------------------------
        A = 0  # exciting static magnetic field strength amplitude in A/m
        A_increment = 10  # excitation step size in A/m
        pv_reached = False
        while not pv_reached:
            pv_mag_static = f_pv_mag(f, mu_h(abs(A)).imag, np.abs(A))

            if abs(pv_limit - pv_mag_static) / pv_limit < max_dev:
                # print(f"{pv_limit = }")
                # print(f"{pv_mag_static = }")
                B_static = mu_h(abs(A)) * A
                # no integral is needed here, because field is constant:
                complex_flux_static = B_static * np.pi * R ** 2  # alt.: integrate_2d_axi_symmetry_field(r_, B_static)

                PF_static_f_op = 2 * np.pi * f * abs(complex_flux_static)
                b_mean_static_f_op = abs(complex_flux_static) / np.pi / R ** 2

                pv_reached = True

            if (pv_mag_static < pv_limit) and (A_increment > 0):
                A = A + A_increment
            elif (pv_mag_static > pv_limit) and (A_increment < 0):
                A = A + A_increment
            elif (pv_mag_static > pv_limit) and (A_increment > 0):
                A_increment = -A_increment / 2
                A = A + A_increment
            else:
                A_increment = -A_increment / 2
                A = A + A_increment

        PF_static_f.append(PF_static_f_op)
        b_mean_static_f.append(b_mean_static_f_op)
        print(f"{A = } (static)")

        # -------------------------
        # IC model
        # -------------------------
        # init permeability with static result:
        r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_h(abs(A)))
        complex_flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)

        A = 1  # exciting static magnetic field strength amplitude in A/m
        A_increment = 10  # excitation step size in A/m
        pv_reached = False
        while not pv_reached:
            # print(f"{A = }")

            r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_h(np.mean(abs(H_))))
            complex_flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)

            pv_mag_ic = f_pv_mag(f, mu_h(abs(H_)).imag, np.abs(H_))
            pv_el_ic = f_pv_el(f, eps.imag, np.abs(E_))
            mean_pv_ic = (integrate_2d_axi_symmetry_field(r_, pv_mag_ic) +
                          integrate_2d_axi_symmetry_field(r_, pv_el_ic)) / (np.pi * R ** 2)

            if abs(pv_limit - mean_pv_ic) / pv_limit < max_dev:
                # print(f"{pv_limit = }")
                # print(f"{pv_mag_static = }")

                # -------------------------
                # Post-Processing
                # -------------------------
                PF_dim_f_op = 2 * np.pi * f * abs(complex_flux_ic)
                b_mean_dim_f_op = abs(complex_flux_ic) / np.pi / R ** 2

                pv_reached = True

            if (mean_pv_ic < pv_limit) and (A_increment > 0):
                A = A + A_increment
            elif (mean_pv_ic > pv_limit) and (A_increment < 0):
                A = A + A_increment
            elif (mean_pv_ic > pv_limit) and (A_increment > 0):
                A_increment = -A_increment / 2
                A = A + A_increment
            else:
                A_increment = -A_increment / 2
                A = A + A_increment

        print(f"{A = } (IC)")

        PF_dim_f.append(PF_dim_f_op)
        b_mean_dim_f.append(b_mean_dim_f_op)
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
