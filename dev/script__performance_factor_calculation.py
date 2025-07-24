import os.path

import matplotlib
import numpy as np

from utils import infinite_cylinder as ic
from utils import material_processing as materials
from utils import comsol_interface as comsol
from utils.physics import *
from utils.maths import *
from utils.general_functions import save_dict
from matplotlib import pyplot as plt
from meta.plot_settings import colors
from meta import paths
import logging
import materialdatabase as mdb

# configure logging to show femmt terminal output
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# -------------------------
# Problem definition
# -------------------------
result_folder = "performance_factor"
R = 7.5e-3  # radius
T_c = 100  # temperature
# T_c = 50  # temperature

f_ = np.array([1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5])
# f_ = np.array([3e5, 4e5, 5e5, 6e5, 7e5, 8e5])
# f_ = np.array([9e5, 1e6])
R_ = [0.004, 0.006, 0.0075, 0.01]

pv_limit = 300000
PF_tdk = []

PF_dim = []
b_mean_dim = []

PF_static = []
b_mean_static = []


# todo: iterate over materials
# -------------------------
# Material (nonlinear)
# -------------------------
# init a material database instance
# mdb_data = mdb.Data()
# complex_permeability = mdb_data.get_complex_permeability(material=mdb.Material.N49,
#                                                          measurement_setup=mdb.MeasurementSetup.TDK_MDT,
#                                                          pv_fit_function=mdb.FitFunction.enhancedSteinmetz)
# print(f"\nExemplary complex permeability data: \n {complex_permeability.measurement_data} \n")
#
# complex_permittivity = mdb_data.get_complex_permittivity(material=mdb.Material.N49,
#                                                          measurement_setup=mdb.MeasurementSetup.LEA_MTB)
# print(f"\nExemplary complex permittivity data: \n {complex_permittivity.measurement_data} \n ")
#
# df_mu = complex_permeability.measurement_data
# df_eps = complex_permittivity.measurement_data

# old material data:
df_mu = materials.read_permeability_txt2df(material_name="N49")
df_eps = materials.read_permittivity_txt2df(material_name="N49")

# -------------------------
# Performance Factor
# -------------------------
# print(df_mu[(df_mu["f"] == 100000) & (df_mu["T"] == 50)])
# b_mean_goal = 50e-3  # Tesla
# flux_goal = b_mean_goal * np.pi * R ** 2


for i, R in enumerate(R_):
    print(f"\n{R = }\n")
    r_ = np.linspace(0, R, 20)  # this is duplicated as a workaround for the static case with the static
    PF_dim_f = []
    b_mean_dim_f = []
    PF_static_f = []
    b_mean_static_f = []
    for f in f_:
        eps = materials.eps_from_df(df_eps, f, T_c)  # todo: here: use the new mdb functionality instead
        mu_h = materials.mu_h_from_df(df_mu, f, T_c)  # todo: here: use the new mdb functionality instead

        # -------------------------
        # Static model
        # -------------------------
        A = 0  # exciting static magnetic field strength amplitude in A/m
        A_increment = 0.05  # excitation step size in A/m
        pv_reached = False
        while not pv_reached:
            A = A + A_increment

            pv_mag_static = f_pv_mag(f, mu_h(abs(A)).imag, np.abs(A))
            # mean_pv_static = integrate_2d_axi_symmetry_field(r_, pv_mag_static) / (np.pi * R ** 2)
            if pv_mag_static > pv_limit:
                pv_reached = True
            B_static = mu_h(abs(A)) * A
            # no integral is needed here, because field is constant:
            # complex_flux_static = integrate_2d_axi_symmetry_field(r_, np.ones_like(r_)*B_static)
            complex_flux_static = B_static * np.pi * R ** 2
            PF_static_f_op = abs(complex_flux_static) * f
            b_mean_static_f_op = abs(complex_flux_static) / np.pi / R ** 2

        PF_static_f.append(PF_static_f_op)
        b_mean_static_f.append(b_mean_static_f_op)
        print(f"{A = } (static)")

        # -------------------------
        # IC model
        # -------------------------
        # init permeability with static result:
        r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_h(abs(A)))
        complex_flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)
        # plt.plot(r_, H_)
        # plt.show()

        A = 0  # exciting static magnetic field strength amplitude in A/m
        pv_reached = False
        while not pv_reached:
            A = A + A_increment
            # print(f"{A = } (IC)")
            # print(f"Flux IC: {flux_ic}")

            r_, H_, E_ = ic.r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_h(np.mean(abs(H_))))
            complex_flux_ic = integrate_2d_axi_symmetry_field(r_, mu_h(abs(H_)) * H_)

            # -------------------------
            # Post-Processing
            # -------------------------
            pv_mag_ic = f_pv_mag(f, mu_h(abs(H_)).imag, np.abs(H_))
            pv_el_ic = f_pv_el(f, eps.imag, np.abs(E_))
            mean_pv_ic = (integrate_2d_axi_symmetry_field(r_, pv_mag_ic) +
                          integrate_2d_axi_symmetry_field(r_, pv_el_ic)) / (np.pi * R ** 2)

            # print(f"{mean_pv_ic = } (IC)")
            # print(f"{A = } (IC)")
            # print(f"magnetic_flux = {np.round(abs(flux_ic) * 1e6, 3)} µVs (from IC)")
            # print(f"mean mag. loss density: {integrate_2d_axi_symmetry_field(r_, pv_mag_ic) / (np.pi * R ** 2) / 1000} kW/m³")
            # print(f"mean el. loss density: {integrate_2d_axi_symmetry_field(r_, pv_el_ic) / (np.pi * R ** 2) / 1000} kW/m³")
            # print(f"mean loss density: {mean_pv_ic / 1000} kW/m³")

            if mean_pv_ic > pv_limit:
                pv_reached = True
            PF_dim_f_op = abs(complex_flux_ic) * f
            b_mean_dim_f_op = abs(complex_flux_ic) / np.pi / R ** 2
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
