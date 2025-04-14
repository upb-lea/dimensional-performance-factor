import numpy as np
from scipy.integrate import trapezoid
from scipy import special
from utils.physics import *
from utils.maths import *


def r_h_e_(R, f, A, eps: complex, mu_h, n_runs=10):
    """

    :param R:
    :param f:
    :param A:
    :param eps: complex permittivity (not the relative permittivity)
    :param mu: complex permeability (not the relative permeability)
    :param n_runs:
    :return:
    """
    permittivity = eps
    r_ = np.linspace(0, R)
    H_LF = np.ones_like(r_) * A
    H_ = H_LF
    runs = np.arange(0, n_runs)
    for run in runs:
        permeability = mu_h(H_)
        k_ = 2 * np.pi * f * np.sqrt(permeability * permittivity)
        H_ = A * special.jv(0, k_ * r_) / special.jv(0, k_ * R)
        E_ = A * k_ / (j * 2 * np.pi * f * permittivity) * special.jv(1, k_ * r_) / special.jv(0, k_ * R)
    return r_, H_, E_


def flux_from_b_(r_, B_):
    return complex(trapezoid(2 * np.pi * r_ * B_.real, dx=r_[1]), trapezoid(2 * np.pi * r_ * B_.imag, dx=r_[1]))
