#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

# Fast weighted max norm matrix completion (with a modified proximal operator)
# using AdaGrad
# Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
import numpy as np
from numpy.math cimport INFINITY
cimport numpy as np
from libc.math cimport sqrt, fabs


def weighted_max_norm_helper_adagrad(
        np.ndarray[np.int64_t, ndim=1] X_rows,
        np.ndarray[np.int64_t, ndim=1] X_cols,
        np.ndarray[np.float32_t, ndim=1] X_values,
        np.ndarray[np.float32_t, ndim=1] weights,
        np.ndarray[np.float32_t, ndim=2] U,
        np.ndarray[np.float32_t, ndim=2] V,
        float sqrt_R, float alpha, float lr, int max_n_epochs, int patience,
        float loss_tol, int verbose):
    cdef np.ndarray[np.float32_t, ndim=2] \
        U_argmin, U_grad, U_sum_grad_sq, V_argmin, V_grad, V_sum_grad_sq, diff
    cdef int m, n, r, wait_counter, epoch_idx, nz_idx, row, col, idx, t, nnz
    cdef float loss, prev_loss, min_loss, val, val2

    m = U.shape[0]
    r = U.shape[1]
    n = V.shape[0]

    nnz = X_rows.shape[0]

    prev_loss = INFINITY
    min_loss = INFINITY
    wait_counter = 0

    U_grad = np.zeros((m, r), dtype=np.float32)
    V_grad = np.zeros((n, r), dtype=np.float32)
    U_sum_grad_sq = np.zeros((m, r), dtype=np.float32)
    V_sum_grad_sq = np.zeros((n, r), dtype=np.float32)
    U_argmin = np.zeros((m, r), dtype=np.float32)
    V_argmin = np.zeros((n, r), dtype=np.float32)
    diff = np.zeros((m, n), dtype=np.float32)
    for epoch_idx in range(max_n_epochs):
        # zero out gradient
        for row in range(m):
            for idx in range(r):
                U_grad[row, idx] = 0
        for col in range(n):
            for idx in range(r):
                V_grad[col, idx] = 0

        # compute gradient
        for nz_idx in range(nnz):
            row = X_rows[nz_idx]
            col = X_cols[nz_idx]

            val = 0
            for idx in range(r):
                val += U[row, idx] * V[col, idx]

            diff[row, col] = \
                (val - X_values[nz_idx]) * weights[nz_idx] / nnz

        for row in range(m):
            for col in range(n):
                for idx in range(r):
                    U_grad[row, idx] += diff[row, col] * V[col, idx]

        for col in range(n):
            for row in range(m):
                for idx in range(r):
                    V_grad[col, idx] += diff[row, col] * U[row, idx]

        # optimizer step
        for row in range(m):
            for idx in range(r):
                val = U_grad[row, idx]
                U_sum_grad_sq[row, idx] += val * val
                U[row, idx] -= lr / sqrt(U_sum_grad_sq[row, idx] + 1e-8) \
                    * val
        for col in range(n):
            for idx in range(r):
                val = V_grad[col, idx]
                V_sum_grad_sq[col, idx] += val * val
                V[col, idx] -= lr / sqrt(V_sum_grad_sq[col, idx] + 1e-8) \
                    * val

        # projection
        for row in range(m):
            val2 = 0
            for idx in range(r):
                val = U[row, idx]
                val2 += val * val
            val2 = sqrt(val2)
            if val2 > sqrt_R:
                val = sqrt_R / val2
                for idx in range(r):
                    U[row, idx] *= val

        for col in range(n):
            val2 = 0
            for idx in range(r):
                val = V[col, idx]
                val2 += val * val
            val2 = sqrt(val2)
            if val2 > sqrt_R:
                val = sqrt_R / val2
                for idx in range(r):
                    V[col, idx] *= val

        # # original entrywise max-norm projection (replaced with version
        # # alternating between scaling rows of U and scaling rows of V
        # # depending on `epoch_idx`)
        # val2 = 0
        # for col in range(n):
        #     for row in range(m):
        #         val = 0
        #         for idx in range(r):
        #             val += U[row, idx] * V[col, idx]
        #         val = fabs(val)
        #         if val > val2:
        #             val2 = val
        # if val2 > alpha:
        #     val2 = sqrt(alpha / val2)
        #     for row in range(m):
        #         for idx in range(r):
        #             U[row, idx] *= val2
        #     for col in range(n):
        #         for idx in range(r):
        #             V[row, idx] *= val2

        # (alternate between rescaling U or rescaling V)
        if epoch_idx % 2 == 0:
            for row in range(m):
                val2 = 0
                for col in range(n):
                    val = 0
                    for idx in range(r):
                        val += U[row, idx] * V[col, idx]
                    val = fabs(val)
                    if val > val2:
                        val2 = val
                if val2 > alpha:
                    val = alpha / val2
                    for idx in range(r):
                        U[row, idx] *= val
        else:
            for col in range(n):
                val2 = 0
                for row in range(m):
                    val = 0
                    for idx in range(r):
                        val += U[row, idx] * V[col, idx]
                    val = fabs(val)
                    if val > val2:
                        val2 = val
                if val2 > alpha:
                    val = alpha / val2
                    for idx in range(r):
                        V[col, idx] *= val

        loss = 0.

        for nz_idx in range(nnz):
            row = X_rows[nz_idx]
            col = X_cols[nz_idx]

            val = 0
            for idx in range(r):
                val += U[row, idx] * V[col, idx]

            val2 = val - X_values[nz_idx]
            loss += val2 * val2 * weights[nz_idx]

        loss /= nnz

        if verbose > 0:
            print('Epoch %d, loss %f' % (epoch_idx + 1, loss), flush=True)

        if prev_loss - loss < loss_tol:
            break

        prev_loss = loss

        if patience > 0:
            if loss < min_loss:
                min_loss = loss
                for row in range(m):
                    for idx in range(r):
                        U_argmin[row, idx] = U[row, idx]
                for col in range(n):
                    for idx in range(r):
                        V_argmin[col, idx] = V[col, idx]
                wait_counter = 0
            else:
                wait_counter += 1
                if wait_counter >= patience:
                    break

    if patience > 0:
        return U_argmin, V_argmin

    return U, V
