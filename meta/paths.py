import json
import os


with open(os.path.join(os.path.dirname(__file__), 'folders.json'), 'r') as file:
    folder_links = json.load(file)

material_data = folder_links["material_data"]
simulation_data = folder_links["simulation_data"]
measurement_data = folder_links["measurement_data"]
grafics = folder_links["grafics"]
