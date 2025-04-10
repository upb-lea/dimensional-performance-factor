import numpy as np

def f_pv_mag(f, mu_imag, h_abs):
    return -0.5 * 2 * np.pi * f * mu_imag * h_abs ** 2


def f_pv_el(f, eps_imag, e_abs):
    return -0.5 * 2 * np.pi * f * eps_imag * e_abs ** 2
