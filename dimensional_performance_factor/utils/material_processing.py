import pandas as pd
from dimensional_performance_factor.meta import paths
from scipy.optimize import curve_fit
from dimensional_performance_factor.utils.physics import *
from dimensional_performance_factor.utils.maths import *
from scipy.interpolate import interp1d, griddata


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


def dielectric_fit_TP(fT, alpha, beta, k, k_f, k_T, k_alpha2):
    f, T = fT
    return (k + k_f / f + k_T * T ** k_alpha2) * T ** alpha * f ** beta


def eps_from_df(df_eps, f, T):
    p_eps_real, cov_eps_real = curve_fit(dielectric_fit_TP, (df_eps["f"], df_eps["T"]), df_eps["eps_real"], maxfev=10000)
    p_eps_imag, cov_eps_imag = curve_fit(dielectric_fit_TP, (df_eps["f"], df_eps["T"]), df_eps["eps_imag"], maxfev=10000)
    eps_real = dielectric_fit_TP((f, T), *p_eps_real)
    eps_imag = dielectric_fit_TP((f, T), *p_eps_imag)
    return epsilon_0 * (eps_real - complex(0, 1) * eps_imag)


def mu_complex_T_f_b(df_mu, T, f, b):
    """
    Linear Interpolation of a permeability dataframe to obtain a flux dependent permeability in a (T,f)-operation point.
    :param df_mu: contains relative permeability
    :param T: scalar
    :param f:scalar
    :param b: vector
    :return: mu(b)
    """
    # Create the grid for interpolation
    grid_f, grid_T, grid_b = np.meshgrid(f, T, b)

    # Factor to rescale  between 0 and 1 for avoiding numerical issues with the meshing
    fac_b = 1/max(b)
    fac_T = 1/T
    fac_f = 1/f

    # Interpolate mu_real values on the grid
    mu_real = griddata((df_mu['f']*fac_f, df_mu['T']*fac_T, df_mu['b']*fac_b), df_mu['mu_real'],
                       (grid_f*fac_f, grid_T*fac_T, grid_b*fac_b), method='linear')

    # Interpolate mu_imag values on the grid
    mu_imag = griddata((df_mu['f']*fac_f, df_mu['T']*fac_T, df_mu['b']*fac_b), df_mu['mu_imag'],
                       (grid_f*fac_f, grid_T*fac_T, grid_b*fac_b), method='linear')

    # return mu_real, mu_imag
    return mu_0*(mu_real[0][0] - complex(0, 1) * mu_imag[0][0])


def mu_complex_h(b, mu_b_complex):
    """
    Converts a flux-dependent permeability mu(b) into a field-dependent permeability mu(h).
    :param b:
    :param mu_b_complex: mu(b)
    :return: h_max, mu(h)
    """
    interpolation_type = "linear"
    # Interpolation with b:
    mu_real_interp_b = interp1d(b, mu_b_complex.real, kind=interpolation_type, fill_value='extrapolate')
    mu_imag_interp_b = interp1d(b, mu_b_complex.imag, kind=interpolation_type, fill_value='extrapolate')

    # Interpolation with h:
    h = b / np.sqrt(mu_real_interp_b(b) ** 2 + mu_imag_interp_b(b) ** 2)
    mu_real_interp_h = interp1d(h, mu_b_complex.real, kind=interpolation_type, fill_value='extrapolate')
    mu_imag_interp_h = interp1d(h, mu_b_complex.imag, kind=interpolation_type, fill_value='extrapolate')

    return max(h), mu_real_interp_h, mu_imag_interp_h


def mu_h_from_df(df_mu, f, T):
    set_b = sorted(df_mu['b'].unique())
    # print(f"b: {set_b}")
    b_extraction = np.linspace(min(set_b), max(set_b), 100)
    mu_b_complex = mu_complex_T_f_b(df_mu=df_mu, T=T, f=f, b=b_extraction)
    h_max, mu_real_interp_h, mu_imag_interp_h = mu_complex_h(b_extraction, mu_b_complex)
    # print(f"h_max: {h_max}")

    def mu_of_h(h):
        return mu_real_interp_h(h) + j * mu_imag_interp_h(h)

    return mu_of_h


def mu_h_from_mu_b(mu_b_complex, b_common):
    h_max, mu_real_interp_h, mu_imag_interp_h = mu_complex_h(b_common, mu_b_complex)

    def mu_of_h(h):
        return mu_real_interp_h(h) + j * mu_imag_interp_h(h)

    return mu_of_h
