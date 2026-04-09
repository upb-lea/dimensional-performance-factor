from utils import infinite_cylinder as ic
from utils import material_processing as material_functions
from utils.physics import *
from utils.maths import *
from utils.general_functions import save_dict
import logging
import materialdatabase as mdb
import numpy as np

# -------------------------
# Logging
# -------------------------
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# -------------------------
# Problem definition
# -------------------------
T_c = 70  # # temperature in C
pv_limit = 300000  # loss density in W/m^3

materials = [
    mdb.Material._3F46,
    mdb.Material.N49,
    mdb.Material.N95,
]

probe_codes_per_material = {
    mdb.Material.N95: ["R29.5x19x14.9"],
    mdb.Material.N49: ["R16x9.6x6.3"],
    mdb.Material._3F46: ["R25x15x10"]
}

R_ = [0.0044, 0.0075, 0.01, 0.013]   # radii in m
core_type = ["PQ20", "PQ40", "PQ50", "PQ65"]   # radii in m

# Example structure: {material: {R: frequency_array}}
# frequency in Hz
frequency_resolution = 30
freqs_per_material = {
    mdb.Material._3F46: {
        0.0044: np.linspace(1e5, 3e6, frequency_resolution),
        0.0075: np.linspace(1e5, 2e6, frequency_resolution),
        0.01: np.linspace(1e5, 1.4e6, frequency_resolution),
        0.013: np.linspace(1e5, 1.1e6, frequency_resolution)
    },
    mdb.Material.N49: {
        0.0044: np.linspace(1e5, 1e6, frequency_resolution),
        0.0075: np.linspace(1e5, 9e5, frequency_resolution),
        0.01: np.linspace(1e5, 8e5, frequency_resolution),
        0.013: np.linspace(1e5, 6e5, frequency_resolution)
    },
    mdb.Material.N95: {
        0.0044: np.linspace(1e5, 9e5, frequency_resolution),
        0.0075: np.linspace(1e5, 7e5, frequency_resolution),
        0.01: np.linspace(1e5, 5e5, frequency_resolution),
        0.013: np.linspace(1e5, 4e5, frequency_resolution)
    },
}


PF_dim = []        # shape: [n_materials][n_radii][n_freqs]
b_mean_dim = []
PF_static = []
b_mean_static = []

# -------------------------
# Material (nonlinear)
# -------------------------
mdb_data = mdb.Data()

# -------------------------
# Performance Factor loop
# -------------------------
for material in materials:
    print(f"\nProcessing {material = }")

    # Prepare material models ...
    complex_permeability = mdb_data.get_complex_permeability(
        material=material,
        data_source=mdb.DataSource.LEA_MTB,
        pv_fit_function=mdb.FitFunction.enhancedSteinmetz,
        probe_codes=probe_codes_per_material[material]
    )
    complex_permeability.fit_losses()
    complex_permeability.fit_permeability_magnitude()
    b_common = np.linspace(0, 0.2, 50)

    complex_permittivity = mdb_data.get_complex_permittivity(
        material=material,
        data_source=mdb.DataSource.LEA_MTB
    )
    complex_permittivity.fit_permittivity_magnitude()
    complex_permittivity.fit_loss_angle()

    PF_dim_r = []
    b_mean_dim_r = []
    PF_static_r = []
    b_mean_static_r = []
    label = []

    for R in R_:
        f_ = freqs_per_material[material][R]  # <-- per-material, per-radius frequency sweep
        print(f"  radius = {R}, frequencies = {f_}")

        PF_dim_f = []
        b_mean_dim_f = []
        PF_static_f = []
        b_mean_static_f = []

        for f in f_:
            print(f"    {f = }")

            # -------------------------
            # Material fit at temperature and frequency
            # -------------------------
            # Permeability:
            mu_real, mu_imag = complex_permeability.fit_real_and_imaginary_part_at_f_and_T(
                f_op=f, T_op=T_c, b_vals=np.linspace(0, 0.2, 50)
            )
            # an interpolation in terms of the magnetic field h is needed:
            mu_h = material_functions.mu_h_from_mu_b((mu_real - j * mu_imag) * mu_0, b_common)

            # Permittivity:
            eps_real, eps_imag = complex_permittivity.fit_real_and_imaginary_part_at_f_and_T(f=f, T=T_c)
            eps = epsilon_0 * (eps_real - complex(0, 1) * eps_imag)

            # -------------------------
            # Static model
            # -------------------------
            flux_static = ic.flux_from_pv__non_linear(pv_limit, R, f, epsilon_0, mu_h)
            PF_static_f.append(2 * np.pi * f * abs(flux_static))
            b_mean_static_f.append(abs(flux_static) / np.pi / R**2)

            # -------------------------
            # IC model
            # -------------------------
            flux_ic = ic.flux_from_pv__non_linear(pv_limit, R, f, eps, mu_h)
            PF_dim_f.append(2 * np.pi * f * abs(flux_ic))
            b_mean_dim_f.append(abs(flux_ic) / np.pi / R**2)

        # append results for this radius
        PF_static_r.append(PF_static_f)
        b_mean_static_r.append(b_mean_static_f)
        PF_dim_r.append(PF_dim_f)
        b_mean_dim_r.append(b_mean_dim_f)

    # append results for this material
    PF_static.append(PF_static_r)
    b_mean_static.append(b_mean_static_r)
    PF_dim.append(PF_dim_r)
    b_mean_dim.append(b_mean_dim_r)

# -------------------------
# Store Results
# -------------------------
results = {
    "temperature": T_c,
    "pv_limit": pv_limit,
    "materials": {}
}

for m_idx, mat in enumerate(materials):
    mat_name = mat.value  # e.g. "N49"
    results["materials"][mat_name] = {}

    for r_idx, R in enumerate(R_):
        results["materials"][mat_name][R] = {
            "core_type": core_type[r_idx],  # store core type
            "frequencies": list(freqs_per_material[mat][R]),  # <-- per-radius array
            "PF_dim": list(PF_dim[m_idx][r_idx]),
            "b_mean_dim": list(b_mean_dim[m_idx][r_idx]),
            "PF_static": list(PF_static[m_idx][r_idx]),
            "b_mean_static": list(b_mean_static[m_idx][r_idx]),
        }

save_dict("PF_comparison.json", results)
