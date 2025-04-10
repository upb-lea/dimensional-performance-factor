from utils import material_processing as materials

# permeability data
df_mu = materials.read_permeability_txt2df(material_name="N49")
print(df_mu)

# permittivity data
df_eps = materials.read_permittivity_txt2df(material_name="N49")
print(df_eps)
