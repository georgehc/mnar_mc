import numpy as np
cimport numpy as np


def shrinkage_singular_values(np.ndarray[np.float64_t, ndim=1] singular_vals,
                              double sum_constraint, double eps):
    cdef double left, center, right, nuc_norm_center
    cdef np.ndarray[np.float64_t, ndim=1] projected_singular_vals
    left = 0.
    right = singular_vals.max()

    while right - left > eps:
        center = (left + right) / 2.
        projected_singular_vals = np.maximum(singular_vals - center, 0.)
        nuc_norm_center = projected_singular_vals.sum()
        if abs(nuc_norm_center - sum_constraint) < eps:
            break
        if nuc_norm_center > sum_constraint:
            left = center
        else:
            right = center

    return np.maximum(singular_vals - right, 0.)
