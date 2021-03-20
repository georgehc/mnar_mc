#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False

# Fast fully-observed 1bitMC propensity score estimation using AdaGrad
# Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
import numpy as np
from numpy.math cimport INFINITY
cimport numpy as np
from libc.math cimport exp, log, sqrt, fabs


def one_bit_MC_fully_observed_approx_helper_adagrad(
        np.ndarray[np.uint8_t, ndim=2] M,
        np.ndarray[np.float32_t, ndim=2] U,
        np.ndarray[np.float32_t, ndim=2] V,
        int rank, float gamma, float lr, int max_n_epochs, int patience,
        float loss_tol, int verbose, int random_seed):
    cdef np.ndarray[np.float32_t, ndim=2] \
        U_argmin, U_grad, U_sum_grad_sq, V_argmin, V_grad, V_sum_grad_sq
    cdef np.ndarray[np.int64_t, ndim=1] random_col_offset, random_col_ordering
    cdef int m, n, r, wait_counter, epoch_idx, batch_idx, row, col, idx, t
    cdef float loss, prev_loss, min_loss, val, val2

    m = U.shape[0]
    r = U.shape[1]
    n = V.shape[0]

    prev_loss = INFINITY
    min_loss = INFINITY
    wait_counter = 0

    U_grad = np.zeros((m, r), dtype=np.float32)
    V_grad = np.zeros((n, r), dtype=np.float32)
    U_sum_grad_sq = np.zeros((m, r), dtype=np.float32)
    V_sum_grad_sq = np.zeros((n, r), dtype=np.float32)
    U_argmin = np.zeros((m, r), dtype=np.float32)
    V_argmin = np.zeros((n, r), dtype=np.float32)
    t = 1
    rng = np.random.RandomState(random_seed)
    for epoch_idx in range(max_n_epochs):
        random_col_offset = rng.permutation(n)
        random_col_ordering = rng.permutation(n)

        for batch_idx in range(n):
            # zero out gradient
            for row in range(m):
                for idx in range(r):
                    U_grad[row, idx] = 0
            for col in range(n):
                for idx in range(r):
                    V_grad[col, idx] = 0

            # compute gradient
            for row in range(m):
                col = random_col_ordering[(random_col_offset[batch_idx]
                                           + row) % n]

                val = 0
                for idx in range(r):
                    val += U[row, idx] * V[col, idx]

                if M[row, col] == 1:
                    val2 = -(1. + exp(val)) * m
                    for idx in range(r):
                        U_grad[row, idx] = V[col, idx] / val2
                        V_grad[col, idx] = U[row, idx] / val2
                else:
                    val2 = (1. + exp(-val)) * m
                    for idx in range(r):
                        U_grad[row, idx] = V[col, idx] / val2
                        V_grad[col, idx] = U[row, idx] / val2

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
            # (alternate between rescaling U or rescaling V)
            if t % 2 == 0:
                for col in range(n):
                    val2 = 0
                    for row in range(m):
                        val = 0
                        for idx in range(r):
                            val += U[row, idx] * V[col, idx]
                        val = fabs(val)
                        if val > val2:
                            val2 = val
                    if val2 > gamma:
                        val = gamma / val2
                        for idx in range(r):
                            V[col, idx] *= val
            else:
                for row in range(m):
                    val2 = 0
                    for col in range(n):
                        val = 0
                        for idx in range(r):
                            val += U[row, idx] * V[col, idx]
                        val = fabs(val)
                        if val > val2:
                            val2 = val
                    if val2 > gamma:
                        val = gamma / val2
                        for idx in range(r):
                            V[row, idx] *= val
            t += 1

        if verbose > 0 or patience > 0:
            loss = 0.
            for row in range(m):
                for col in range(n):
                    val = 0
                    for idx in range(r):
                        val += U[row, idx] * V[col, idx]

                    if M[row, col] == 1:
                        loss -= log(1. / (1. + exp(-val)))
                    else:
                        loss -= log(1. / (1. + exp(val)))
            loss /= m*n

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
