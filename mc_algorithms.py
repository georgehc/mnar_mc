"""
Implementations of some matrix completion algorithms

We've implemented two matrix completion algorithms meant for estimating
propensity scores from fully observed binary matrices:
- 1-bit matrix completion (Davenport et al 2014) specialized to fully-observed
  data; we refer to this algorithm as "1bitMC" (with some variants on what gets
  capitalized and whether the "1" at the front gets replaced by "one" due to
  Python not allowing variable names to start with numbers)
- a modified version of 1-bit matrix completion meant for handling propensity
  scores that are exactly 1 (we call this algorithm 1bitMC-mod); this is
  discussed in the longer version of the paper where we use it in conjunction
  with a modified logistic regression link function

We've implemented the following matrix completion algorithms and their variants
that can weight the different matrix entries in the squared loss for ratings
(the idea is to inversely weight based on estimated propensity scores):
- weighted SoftImpute: this solves an optimization problem like the one solved
  by SoftImpute (squared error loss for ratings paired with nuclear norm
  regularization on estimated ratings matrix) except that the entries in the
  squared error loss are weighted; note that how this convex program is solved
  is completely different from the SoftImpute algorithm (namely, we just use
  proximal gradient)
- weighted-trace-norm-regularized matrix completion (Srebro and Salakhutdinov
  2010), where the squared error loss on the ratings can further be weighted
  (rather than only the regularization being weighted); we refer to this
  algorithm as doubly-weighted trace norm
- max-norm-constrained matrix completion (Cai and Zhou 2016); we refer to this
  algorithm as MaxNorm; the vanilla version does not have weights on entries
  in the squared loss for ratings
- a variant of MaxNorm where the entries in the squared loss for ratings are
  weighted

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (weima@cmu.edu)

This code uses the following library (on top of Anaconda Python 3.7):
- copt: pip install -U copt

References:

    T. Tony Cai and Wen-Xin Zhou. Matrix completion via max-norm constrained
    optimization. Electronic Journal of Statistics, 10(1):1493–1525, 2016.

    Mark A. Davenport, Yaniv Plan, Ewout Van Den Berg, and Mary Wootters.
    1-bit matrix completion. Information and Inference, 3(3):189–223, 2014

    Jason D. Lee, Ben Recht, Nathan Srebro, Joel Tropp, and Ruslan R.
    Salakhutdinov. Practical large-scale optimization for max-norm
    regularization. In Advances in Neural Information Processing Systems,
    pages 1297-1305, 2010.

    Nathan Srebro and Ruslan R. Salakhutdinov. Collaborative filtering in a
    non-uniform world: Learning with the weighted trace norm. In Advances in
    Neural Information Processing Systems, pages 2056–2064, 2010.
"""
import numpy as np
import os
from copt import minimize_proximal_gradient, minimize_three_split
from scipy.linalg import svd
from scipy.optimize import minimize
from sklearn.utils.extmath import randomized_svd
from subprocess import DEVNULL, call
from nuc_shrinkage_helper import shrinkage_singular_values

# prevent numpy/scipy/etc from only using a single processor; see:
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
# (note that this is unix/linux only and should silently error on other
# platforms)
call(['taskset', '-p', '0x%s' % ('f' * int(np.ceil(os.cpu_count() / 4))),
      '%d' % os.getpid()], stdout=DEVNULL, stderr=DEVNULL)


def std_logistic_function(x):
    return 1 / (1 + np.exp(-x))


def grad_std_logistic_function(x):
    z = np.exp(-x)
    return z / (1 + z)**2


def mod_logistic_function(x, gamma, one_minus_logistic_gamma):
    x = np.clip(x, -gamma, gamma)
    return 1 / (1 + np.exp(-x)) + .5 * (1 + x/gamma) * one_minus_logistic_gamma


def grad_mod_logistic_function(x, gamma, one_minus_logistic_gamma):
    x = np.clip(x, -gamma, gamma)
    z = np.exp(-x)
    return z / (1 + z)**2 + one_minus_logistic_gamma / (2*gamma)


def one_bit_MC_fully_observed(M, link, link_gradient, tau, gamma,
                              max_rank=None, opt_max_iter=2000,
                              opt_eps=1e-6, shrinkage_search_eps=1e-8,
                              max_num_proj_iter=1000):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m = M.shape[0]
    n = M.shape[1]
    tau_sqrt_mn = tau * np.sqrt(m*n)

    def prox_nuc_norm(_A, t=1):
        # nuclear norm projection
        _A = _A.reshape(m, n)
        if max_rank is None:
            U, S, VT = svd(_A, full_matrices=False)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float64),
                                              tau_sqrt_mn,
                                              shrinkage_search_eps)
                _A = np.dot(U * S, VT)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float64),
                                              tau_sqrt_mn,
                                              shrinkage_search_eps)
            _A = np.dot(U * S, VT)
        return _A.flatten()

    def prox_entrywise_max_norm(_A, t=1):
        return np.clip(_A, -gamma, gamma)

    M_one_mask = (M == 1)
    M_zero_mask = (M == 0)

    def f(_A, return_gradient=True):
        _A = np.clip(_A.reshape(m, n), -gamma, gamma)

        loss = -(np.log(link(_A[M_one_mask])).sum()
                 + np.log(1 - link(_A[M_zero_mask])).sum())

        if not return_gradient:
            return loss

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask])/link(_A[M_one_mask])
        grad[M_zero_mask] = \
            link_gradient(_A[M_zero_mask])/(1 - link(_A[M_zero_mask]))
        return loss, grad.flatten()

    A_hat = minimize_three_split(f, np.zeros(m*n, dtype=np.float64),
                                 prox_entrywise_max_norm, prox_nuc_norm,
                                 max_iter=opt_max_iter, tol=opt_eps).x
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def one_bit_MC_mod_fully_observed(M, link, link_gradient, tau, gamma,
                                  max_rank=None, opt_max_iter=2000,
                                  opt_eps=1e-6, phi=None,
                                  shrinkage_search_eps=1e-8,
                                  max_num_proj_iter=1000):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m = M.shape[0]
    n = M.shape[1]
    tau_sqrt_mn = tau * np.sqrt(m*n)

    if phi is None:
        phi = .95 * gamma

    def prox_nuc_norm(_A, t=1):
        # nuclear norm projection
        _A = _A.reshape(m, n)
        if max_rank is None:
            U, S, VT = svd(_A, full_matrices=False)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float64),
                                              tau_sqrt_mn,
                                              shrinkage_search_eps)
                _A = np.dot(U * S, VT)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float64),
                                              tau_sqrt_mn,
                                              shrinkage_search_eps)
            _A = np.dot(U * S, VT)
        return _A.flatten()

    M_zero_mask = (M == 0)

    def prox_entrywise_max_norm_mod(_A, t=1):
        _A = np.clip(_A, -gamma, gamma).reshape(m, n)
        _A[M_zero_mask] = np.minimum(_A[M_zero_mask], phi)
        return _A.flatten()

    M_one_mask = (M == 1)

    def f(_A, return_gradient=True):
        _A = np.clip(_A.reshape(m, n), -gamma, gamma)
        _A[M_zero_mask] = np.minimum(_A[M_zero_mask], phi)

        loss = -(np.log(link(_A[M_one_mask])).sum()
                 + np.log(1 - link(_A[M_zero_mask])).sum())

        if not return_gradient:
            return loss

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask])/link(_A[M_one_mask])
        grad[M_zero_mask] = \
            link_gradient(_A[M_zero_mask])/(1 - link(_A[M_zero_mask]))
        return loss, grad.flatten()

    A_hat = minimize_three_split(f, np.zeros(m*n, dtype=np.float64),
                                 prox_entrywise_max_norm_mod, prox_nuc_norm,
                                 max_iter=opt_max_iter, tol=opt_eps).x
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def weighted_softimpute(X, M, W, lmbda, min_value=None, max_value=None,
                        max_rank=None, opt_max_iter=100, opt_eps=1e-6):
    # if `max_rank` is set to None, then exact SVD is used
    m = X.shape[0]
    n = X.shape[1]

    def prox(Z, t):
        Z = Z.reshape(m, n)

        # singular value shrinkage
        if max_rank is None:
            U, S, VT = svd(Z, full_matrices=False)
        else:
            U, S, VT = randomized_svd(Z, max_rank)
        S = np.maximum(S - lmbda*t, 0)
        Z = np.dot(U * S, VT)
        return Z.flatten()

    M_one_mask = (M == 1)
    masked_weights = W[M_one_mask]
    masked_X = X[M_one_mask]

    def f(Z, return_gradient=True):
        Z = Z.reshape(m, n)

        diff = (Z[M_one_mask] - masked_X) * np.sqrt(masked_weights)
        loss = 1/2 * np.inner(diff, diff) \
            + lmbda * np.linalg.norm(Z, ord='nuc')

        if not return_gradient:
            return loss

        grad = np.zeros((m, n))
        grad[M_one_mask] = diff
        return loss, grad.flatten()

    if min_value is not None or max_value is not None:
        def min_max_value_proj(Z, t):
            return np.clip(Z, min_value, max_value)
        X_hat = minimize_three_split(f, np.zeros(m*n, dtype=np.float64),
                                     prox, min_max_value_proj,
                                     max_iter=opt_max_iter, tol=opt_eps).x
    else:
        X_hat = minimize_proximal_gradient(f, np.zeros(m*n, dtype=np.float64),
                                           prox, jac=True,
                                           max_iter=opt_max_iter,
                                           tol=opt_eps).x
    return X_hat.reshape(m, n)


def approx_doubly_weighted_trace_norm(X, M, W, n_components, lmbda,
                                      min_value=None, max_value=None,
                                      opt_max_iter=100, opt_eps=1e-6,
                                      random_state=None,
                                      trace_norm_weighting=True):
    m, n = X.shape

    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    if trace_norm_weighting:
        total = float(np.sum(X))
        row_weights = np.sum(X, axis=1) / total * m
        col_weights = np.sum(X, axis=0) / total * n
    else:
        row_weights = np.ones(m)
        col_weights = np.ones(n)

    U = rng.rand(m, n_components)
    V = rng.rand(n, n_components)
    m_times_n_components = m * n_components

    def f(_UV, return_gradient=True):
        U = _UV[:m_times_n_components].reshape(m, n_components)
        V = _UV[m_times_n_components:].reshape(n, n_components)
        X_hat_tmp = U.dot(V.T)
        diff = (X_hat_tmp - X) * M * np.sqrt(W)

        U_weighted = np.dot(np.diag(np.sqrt(row_weights)), U)
        V_weighted = np.dot(np.diag(np.sqrt(col_weights)), V)

        # by default, np.linalg.norm for matrices is Frobenius norm
        loss = (np.linalg.norm(diff)**2
                + lmbda * (np.linalg.norm(U_weighted)**2
                           + np.linalg.norm(V_weighted)**2)) / 2

        if not return_gradient:
            return loss

        grad_U = np.dot(diff, V) + lmbda * U_weighted
        grad_V = np.dot(diff.T, U) + lmbda * V_weighted
        grad_UV = np.concatenate([grad_U.flatten(), grad_V.flatten()])
        return loss, grad_UV

    if min_value is not None or max_value is not None:
        def min_max_value_proj(Z, t):
            return np.clip(Z, min_value, max_value)
        _UV_hat = minimize_proximal_gradient(f,
                                             np.concatenate([U.flatten(),
                                                             V.flatten()]),
                                             min_max_value_proj, jac=True,
                                             max_iter=opt_max_iter,
                                             tol=opt_eps).x
    else:
        _UV_hat = \
            minimize(f, np.concatenate([U.flatten(), V.flatten()]),
                     method='L-BFGS-B', jac=True,
                     options={'maxiter': opt_max_iter},
                     tol=opt_eps).x
    U = _UV_hat[:m_times_n_components].reshape(m, n_components)
    V = _UV_hat[m_times_n_components:].reshape(n, n_components)
    X_hat = U.dot(V.T)
    return X_hat


def weighted_max_norm(X, M, W, n_components=20, opt_eps=1e-6, opt_max_iter=100,
                      init_std=0.01, R=0.01, alpha=5, random_state=None):
    m, n = X.shape
    sqrt_R = np.sqrt(R)

    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    U = init_std * rng.rand(m, n_components)
    V = init_std * rng.rand(n, n_components)
    m_times_n_components = m * n_components

    def f(_UV, return_gradient=True):
        U = _UV[:m_times_n_components].reshape(m, n_components)
        V = _UV[m_times_n_components:].reshape(n, n_components)
        X_hat_tmp = U.dot(V.T)
        diff = (X_hat_tmp - X) * M * np.sqrt(W)
        loss = 1/2 * np.linalg.norm(diff)**2

        if not return_gradient:
            return loss

        grad_U = np.dot(diff, V)
        grad_V = np.dot(U.T, diff).T
        grad_UV = np.concatenate([grad_U.flatten(), grad_V.flatten()])
        return loss, grad_UV

    def prox(_UV, t):
        U_tmp = _UV[:m_times_n_components].reshape(m, n_components)
        V_tmp = _UV[m_times_n_components:].reshape(n, n_components)
        tmp_norm_inf = np.max(np.abs(U_tmp.dot(V_tmp.T)))
        if tmp_norm_inf > alpha:
            U_tmp = U_tmp * np.sqrt(alpha) / np.sqrt(tmp_norm_inf)
            V_tmp = V_tmp * np.sqrt(alpha) / np.sqrt(tmp_norm_inf)
        U = max_row_l2_norm_proj(U_tmp, sqrt_R)
        V = max_row_l2_norm_proj(V_tmp, sqrt_R)
        prox_UV = np.concatenate([U.flatten(), V.flatten()])
        return prox_UV

    _UV_hat = \
        minimize_proximal_gradient(f,
                                   np.concatenate([U.flatten(),
                                                   V.flatten()]),
                                   prox, jac=True, max_iter=opt_max_iter,
                                   tol=opt_eps).x
    U = _UV_hat[:m_times_n_components].reshape(m, n_components)
    V = _UV_hat[m_times_n_components:].reshape(n, n_components)
    X_hat = U.dot(V.T)
    return X_hat


def max_row_l2_norm_proj(U, threshold):
    n, k = U.shape
    for i in range(n):
        row_l2_norm = np.linalg.norm(U[i])
        if row_l2_norm > threshold:
            U[i] = U[i] / row_l2_norm * threshold
    return U
