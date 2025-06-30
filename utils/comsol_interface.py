import pandas as pd
import numpy as np


def read_df_from_comsol_3d_plot(link2file, quantity_name: str):
    df = pd.read_csv(link2file, sep='\s+', comment="%", skiprows=8, header=None)
    df.columns = ["x", "y", "z", quantity_name]
    return df


def read_df_from_comsol_table(link2file, header: list):
    df = pd.read_csv(link2file, sep='\s+', comment="%", skiprows=5, header=None)
    df.columns = header
    return df


def read_comsol_2d_circle_field(link, field_name, R):
    df = read_df_from_comsol_3d_plot(link2file=link, quantity_name=field_name)

    x_field, y_field, scalar_field = df.loc[(np.sqrt(df["x"] ** 2 + df["y"] ** 2) <= R), "x"].to_numpy(), \
                                     df.loc[(np.sqrt(df["x"] ** 2 + df["y"] ** 2) <= R), "y"].to_numpy(), \
                                     df.loc[(np.sqrt(df["x"] ** 2 + df["y"] ** 2) <= R), field_name].to_numpy()
    return x_field, y_field, scalar_field
