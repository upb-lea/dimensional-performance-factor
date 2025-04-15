import numpy as np
from scipy.integrate import trapezoid
from scipy import special
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
from utils.physics import *
from utils.maths import *


def r_h_e_(R, f, A, eps: complex, mu_of_h, n_runs=10):
    """

    :param R:
    :param f:
    :param A:
    :param eps: complex permittivity (not the relative permittivity)
    :param mu_of_h: complex permeability (not the relative permeability)
    :param n_runs:
    :return:
    """
    permittivity = eps
    r_ = np.linspace(0, R, 50)
    H_LF = np.ones_like(r_) * A
    H_ = H_LF
    runs = np.arange(0, n_runs)
    for run in runs:
        permeability = mu_of_h(np.abs(H_))
        print(np.abs(H_))
        print(permeability)
        k_ = 2 * np.pi * f * np.sqrt(permeability * permittivity)
        H_ = A * special.jv(0, k_ * r_) / special.jv(0, k_ * R)
        E_ = A * k_ / (j * 2 * np.pi * f * permittivity) * special.jv(1, k_ * r_) / special.jv(0, k_ * R)
    return r_, H_, E_


def r_h_e_NL(R, f, A, eps: complex, mu_h, n_runs=10):
    """

    :param R:
    :param f:
    :param A:
    :param eps: complex permittivity (not the relative permittivity)
    :param mu: complex permeability (not the relative permeability)
    :param n_runs:
    :return:
    """
    r_ = np.linspace(1e-5, R, 50)

    permittivity = eps
    H_vals_interpolation = np.linspace(0, 70, 200)  # TODO: this should be an argument with regard to the material
    permeability = mu_h(H_vals_interpolation)
    k_at_h = 2 * np.pi * f * np.sqrt(permeability * permittivity)

    k_squared_interp = interp1d(np.abs(H_vals_interpolation), k_at_h**2, kind="cubic", fill_value='extrapolate')

    # Define the ODE system
    def bessel_system(r, y):
        H, dH = y
        k2 = k_squared_interp(np.abs(H))
        d2H = -(1 / r) * dH - (k2 - 1) * H
        d2H[r == 0] = - (k2[r == 0] - 1) * H[r == 0]  # handle r=0
        return np.vstack((dH, d2H))

    def bc(ya, yb):
        return np.array([
            yb[0] - A,  # H(R) = 0.01 (for example)
            ya[1]  # dH/dr at r=0 = 0 (symmetry)
        ])

    H_guess = np.exp(-r_)  # or any smooth function that meets BC roughly
    guess = np.vstack((H_guess, -H_guess))

    # Solve BVP
    sol = solve_bvp(bessel_system, bc, r_, guess)

    H_ = sol.y[0]
    dH_dr = sol.y[1]
    E_ = (1 / (1j * 2 * np.pi * f * eps)) * dH_dr
    return sol.x, H_, E_


def flux_from_b_(r_, B_):
    dr = r_[3] - r_[2]
    # return complex(trapezoid(2 * np.pi * r_ * B_.real, dx=dr), trapezoid(2 * np.pi * r_ * B_.imag, dx=dr))
    # return complex(np.trapz(2 * np.pi * r_ * B_.real, r_), np.trapz(2 * np.pi * B_.imag, r_))
    return np.trapz(2 * np.pi * r_ * B_, r_)
