#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport fabs


def shrinkage_singular_values(np.ndarray[np.float32_t, ndim=1] singular_vals,
                              float sum_constraint, float eps):
    cdef float left, center, right, nuc_norm_center, val
    cdef int length, idx
    cdef np.ndarray[np.float32_t, ndim=1] projected_singular_vals
    length = singular_vals.shape[0]
    left = 0.
    right = singular_vals.max()
    projected_singular_vals = np.zeros(length, dtype=np.float32)

    while right - left > eps:
        center = (left + right) / 2.
        nuc_norm_center = 0
        for idx in range(length):
            val = singular_vals[idx] - center
            if val > 0:
                nuc_norm_center += val
        if fabs(nuc_norm_center - sum_constraint) < eps:
            break
        if nuc_norm_center > sum_constraint:
            left = center
        else:
            right = center

    for idx in range(length):
        val = singular_vals[idx] - right
        if val > 0:
            projected_singular_vals[idx] = val
    return projected_singular_vals
