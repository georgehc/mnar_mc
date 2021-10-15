#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
import numpy as np
cimport numpy as np
from libc.math cimport fabs


def shrinkage_singular_values(np.ndarray[np.float32_t, ndim=1] singular_vals,
                              float sum_constraint):
    cdef float partial_sum, threshold
    cdef int length, idx, best_k, idx_plus_one
    cdef np.ndarray[np.float32_t, ndim=1] projected_singular_vals
    length = singular_vals.shape[0]
    projected_singular_vals = np.zeros(length, dtype=np.float32)

    partial_sum = 0.
    for idx in range(length):
        partial_sum += singular_vals[idx]
        idx_plus_one = idx + 1

        if partial_sum - idx_plus_one * singular_vals[idx] <= sum_constraint:
            best_k = idx_plus_one

    threshold = (singular_vals[:best_k].sum() - sum_constraint) / best_k
    for idx in range(best_k):
        projected_singular_vals[idx] = singular_vals[idx] - threshold

    return projected_singular_vals
