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
T_c = 25  # temperature
pv_limit = 300000

materials = [
    mdb.Material.N49,
    mdb.Material.N95
]

# frequency sweeps per material
frequency_resolution = 30
freqs_per_material = {
    mdb.Material.N49: np.linspace(1e5, 1e6, frequency_resolution),
    mdb.Material.N95: np.linspace(1e5, 1e6, frequency_resolution),
}

# -------------------------
# Material database
# -------------------------
mdb_data = mdb.Data()

# -------------------------
# Results container
# -------------------------
results = {
    "temperature": T_c,
    "pv_limit": pv_limit,
    "materials": {}
}

# -------------------------
# Performance Factor loop
# -------------------------
for material in materials:
    print(f"\nProcessing {material = }")
    mat_name = material.value

    # Prepare material models
    complex_permeability = mdb_data.get_complex_permeability(
        material=material,
        data_source=mdb.DataSource.TDK_MDT,
        pv_fit_function=mdb.FitFunction.enhancedSteinmetz
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

    # Store results for this material
    results["materials"][mat_name] = {
        "frequencies": list(freqs_per_material[material]),
        "PF_static": []
    }

    # Loop frequencies
    for f in freqs_per_material[material]:
        print(f"  {f = }")

        # Material fit
        mu_real, mu_imag = complex_permeability.fit_real_and_imaginary_part_at_f_and_T(
            f_op=f, T_op=T_c, b_vals=b_common
        )
        mu_h = material_functions.mu_h_from_mu_b((mu_real - j * mu_imag) * mu_0, b_common)

        eps_real, eps_imag = complex_permittivity.fit_real_and_imaginary_part_at_f_and_T(f=f, T=T_c)
        eps = epsilon_0 * (eps_real - complex(0, 1) * eps_imag)

        # -------------------------
        # Static model
        # -------------------------
        A = 0
        A_increment = 0.01
        pv_reached = False
        while not pv_reached:
            A += A_increment
            pv_mag_static = f_pv_mag(f, mu_h(abs(A)).imag, np.abs(A))
            if pv_mag_static > pv_limit:
                pv_reached = True
            B_static = mu_h(abs(A)) * A
            complex_flux_static = B_static  # no radius dependency anymore
            PF_static_f_op = f * abs(complex_flux_static)

        results["materials"][mat_name]["PF_static"].append(PF_static_f_op)

# -------------------------
# Store Results
# -------------------------
save_dict("PF_static_comparison.json", results)
