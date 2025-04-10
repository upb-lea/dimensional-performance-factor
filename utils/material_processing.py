import pandas as pd
from meta import paths


def read_permeability_txt2df(material_name):
    """
    :param material_name:
    :return:
    """
    # Read txt file
    df_mu = pd.read_csv(f"{paths.material_data}/{material_name}/comsol/mu_complex.txt", sep=",", header=None)
    df_mu.columns = ["f", "T", "b", "mu_real", "mu_imag"]
    return df_mu


def read_permittivity_txt2df(material_name):
    """
    :param material_name:
    :return:
    """
    # Read txt file
    df_eps = pd.read_csv(f"{paths.material_data}/{material_name}/comsol/eps_complex.txt", sep=",", header=None)
    df_eps.columns = ["f", "T", "eps_real", "eps_imag"]
    return df_eps
