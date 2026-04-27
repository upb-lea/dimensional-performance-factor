"""Load user-defined folder paths from a local 'user_paths.json' file.

This file expects that the user has created a 'user_paths.json' file in the same
directory as this Python file.

The 'user_paths.json' file should define paths to the material data, simulation
data, measurement data, and graphics/plots folders.
"""
import json
import os
import warnings
from pathlib import Path

expected_user_paths_json = """
Expected content of 'user_paths.json':

{
  "material_data": "C:/Users/.../materials",
  "simulation_data": "C:/Users/.../simulations",
  "measurement_data": "C:/Users/.../measurements",
  "grafics": "C:/Users/.../plots"
}
"""

try:
    with open(os.path.join(os.path.dirname(__file__), "user_paths.json"), "r") as file:
        folder_links = json.load(file)

    material_data = Path(folder_links["material_data"])
    simulation_data = Path(folder_links["simulation_data"])
    measurement_data = Path(folder_links["measurement_data"])
    grafics = Path(folder_links["grafics"])

except Exception as error:
    warnings.warn(
        f"Something is wrong with 'user_paths.json'.\n"
        f"Error: {error}\n\n"
        f"{expected_user_paths_json}"
    )
    raise