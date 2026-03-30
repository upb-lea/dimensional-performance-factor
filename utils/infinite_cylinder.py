from scipy import special
from utils.maths import *


def r_h_e_linear(R, f, A, eps: complex, mu: complex, print_wavelength=False):
    """

    :param R:
    :param f:
    :param A:
    :param print_wavelength:
    :param eps: complex permittivity (not the relative permittivity)
    :param mu_of_h: complex permeability (not the relative permeability)
    :param n_runs:
    :return:
    """
    permittivity = eps
    permeability = mu
    r_ = np.linspace(0, R, 20)
    k_ = 2 * np.pi * f * np.sqrt(permeability * permittivity)
    H_ = A * special.jv(0, k_ * r_) / special.jv(0, k_ * R)
    E_ = A * k_ / (j * 2 * np.pi * f * permittivity) * special.jv(1, k_ * r_) / special.jv(0, k_ * R)
    if print_wavelength:
        wavelength = 2*np.pi/k_.real
        print(f"wavelength at {f} Hz: {wavelength*1000} mm")
        print(f"diameter-to-wavelength ratio at {f} Hz: {2*R/wavelength}")
    return r_, H_, E_

