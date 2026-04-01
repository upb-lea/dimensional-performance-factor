from scipy import special
from utils.maths import *
from utils.physics import *


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
        wavelength = 2 * np.pi / k_.real
        print(f"wavelength at {f} Hz: {wavelength * 1000} mm")
        print(f"diameter-to-wavelength ratio at {f} Hz: {2 * R / wavelength}")
    return r_, H_, E_


def pv_from_b_mean__linear(R, f, b_mean, eps, mu, rel_tol=1e-3):
    """
    Compute the mean-cross-sectional core loss density of a cylinder with radius R.

    :param R: radius [m]
    :param f: frequency [Hz]
    :param b_mean: mean-cross-sectional mag. flux density (peak value) [Hz]
    :param eps: constant complex permittivity
    :param mu: constant complex permeability
    :param rel_tol: relative tolerance of the total magnetic flux through the cross-section
    :return: mean-cross-sectional core loss density
    """
    flux_selected = b_mean * np.pi * R ** 2

    # init step
    A = 1
    r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
    flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

    # iterative rescaling
    while abs(flux_selected - abs(flux_ic)) / flux_selected > rel_tol:
        A *= flux_selected / abs(flux_ic)
        r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu)
        flux_ic = integrate_2d_axi_symmetry_field(r_, mu * H_)

    # Rel. Deviation from selected flux
    # print(f"Rel. Deviation from selected flux {abs(flux_selected - abs(flux_ic)) / flux_selected}")

    # mag. loss density over the radius r_
    pv_mag = f_pv_mag(f, mu.imag, np.abs(H_))
    # el. loss density over the radius r_
    pv_el = f_pv_el(f, eps.imag, np.abs(E_))

    # return the mean-cross-sectional core loss density
    return mean_pv(r_, pv_mag, pv_el)


def pv_from_b_mean__non_linear(R, f, b_mean, eps, mu_of_h, rel_tol=1e-6):
    """
    Compute the mean-cross-sectional core loss density of a cylinder with radius R.

    :param R: radius [m]
    :param f: frequency [Hz]
    :param b_mean: mean-cross-sectional mag. flux density (peak value) [Hz]
    :param eps: constant complex permittivity
    :param mu_of_h: flux-dependent complex permeability
    :param rel_tol: relative tolerance of the total magnetic flux through the cross-section
    :return: mean-cross-sectional core loss density
    """
    flux_selected = b_mean * np.pi * R ** 2

    # init step
    # init permeability with static result:
    A = 1
    r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_of_h(abs(A)))
    flux_ic = integrate_2d_axi_symmetry_field(r_, mu_of_h(abs(H_)) * H_)

    # iterative rescaling
    while abs(flux_selected - abs(flux_ic)) / flux_selected > rel_tol:
        A *= flux_selected / abs(flux_ic)
        r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_of_h(abs(A)))
        flux_ic = integrate_2d_axi_symmetry_field(r_, mu_of_h(abs(H_)) * H_)

    # Rel. Deviation from selected flux
    # print(f"Rel. Deviation from selected flux {abs(flux_selected - abs(flux_ic)) / flux_selected}")

    # mag. loss density over the radius r_
    pv_mag = f_pv_mag(f, mu_of_h(abs(H_)).imag, np.abs(H_))
    # el. loss density over the radius r_
    pv_el = f_pv_el(f, eps.imag, np.abs(E_))

    # return the mean-cross-sectional core loss density
    return mean_pv(r_, pv_mag, pv_el)

def mean_pv(r_, pv_mag, pv_el):

    R = max(r_)
    # return the mean-cross-sectional core loss density
    return (integrate_2d_axi_symmetry_field(r_, pv_mag) +
            integrate_2d_axi_symmetry_field(r_, pv_el)) / (np.pi * R ** 2)

def flux_from_pv__non_linear(pv_selected, R, f, eps, mu_of_h, rel_tol=1e-3):
    """
    Compute the mean-cross-sectional core loss density of a cylinder with radius R.

    :param pv: mean-cross-sectional loss density [W/m^3]
    :param R: radius [m]
    :param f: frequency [Hz]
    :param eps: constant complex permittivity
    :param mu_of_h: flux-dependent complex permeability
    :param rel_tol: relative tolerance of the total magnetic flux through the cross-section
    :return: mean-cross-sectional core loss density
    """
    # init step
    A = 5
    r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_of_h(abs(A)))

    # mag. loss density over the radius r_
    pv_mag = f_pv_mag(f, mu_of_h(abs(H_)).imag, np.abs(H_))

    # el. loss density over the radius r_
    pv_el = f_pv_el(f, eps.imag, np.abs(E_))

    # mean total core loss density
    pv_mean = mean_pv(r_, pv_mag, pv_el)

    relax = 0.4
    # iterative rescaling
    while abs(pv_selected - pv_mean) / pv_mean > rel_tol:

        # rescaling with relaxing factor
        A *= np.exp(0.5 * relax * np.log(pv_selected / pv_mean))
        r_, H_, E_ = r_h_e_linear(R=R, f=f, A=A, eps=eps, mu=mu_of_h(abs(A)))

        # mag. loss density over the radius r_
        pv_mag = f_pv_mag(f, mu_of_h(abs(H_)).imag, np.abs(H_))

        # el. loss density over the radius r_
        pv_el = f_pv_el(f, eps.imag, np.abs(E_))

        # mean total core loss density
        pv_mean = mean_pv(r_, pv_mag, pv_el)

    flux_ic = integrate_2d_axi_symmetry_field(r_, mu_of_h(abs(H_)) * H_)

    # Rel. Deviation from loss density
    # print(f"Rel. Deviation from selected loss density {abs(pv_selected - pv_mean) / pv_mean}")

    # return the total magnetic flux through the cross-section
    return flux_ic
