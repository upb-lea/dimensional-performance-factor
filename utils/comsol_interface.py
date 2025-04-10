import pandas as pd


def read_df_from_comsol_3d_plot(link2file, quantity_name: str):
    df = pd.read_csv(link2file, sep='\s+', comment="%", skiprows=8, header=None)
    df.columns = ["x", "y", "z", quantity_name]
    return df

