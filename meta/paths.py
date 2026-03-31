import json
import os
from pathlib import Path

with open(os.path.join(os.path.dirname(__file__), 'folders.json'), 'r') as file:
    folder_links = json.load(file)

material_data = Path(folder_links["material_data"])
simulation_data = Path(folder_links["simulation_data"])
measurement_data = Path(folder_links["measurement_data"])
grafics = Path(folder_links["grafics"])
