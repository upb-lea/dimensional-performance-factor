import json
import os


with open(os.path.join(os.path.dirname(__file__), 'folders.json'), 'r') as file:
    folder_links = json.load(file)

comsol_results = folder_links["comsol_results"]
material_data = folder_links["material_data"]
grafics = folder_links["grafics"]
measurements = folder_links["measurements"]
