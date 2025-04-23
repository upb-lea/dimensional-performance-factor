import numpy as np
from scipy.spatial import Delaunay

j = complex(0, 1)


def area_from_3_points(x, y, z):
    return np.sqrt(np.sum(np.cross(x-y, x-z), axis=-1)**2)/2


def integrate_2d(x, y, f):
    # Create a triangulation
    domain_points = np.array(list(zip(x, y)))
    tri = Delaunay(domain_points)
    int_f = 0
    for vertices in tri.simplices:
        mean_value = (f[vertices[0]] + f[vertices[1]] + f[vertices[2]]) / 3
        area = area_from_3_points(domain_points[vertices[0]], domain_points[vertices[1]], domain_points[vertices[2]])
        int_f += mean_value * area

    return int_f


def integral_2d_axi_symmetry_flux_from_b_(r_, field_):
    return np.trapz(2 * np.pi * r_ * field_, r_)
