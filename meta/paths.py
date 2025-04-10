import json
import os


with open(os.path.join(os.path.dirname(__file__), 'folders.json'), 'r') as file:
    folder_links = json.load(file)

path_comsol_results = folder_links["comsol_results"]
path_material_data = folder_links["material_data"]

