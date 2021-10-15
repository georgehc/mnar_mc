"""
Implementations of some matrix completion algorithms

We've implemented two matrix completion algorithms meant for estimating
propensity scores from fully observed binary matrices:
- 1-bit matrix completion (Davenport et al 2014) specialized to fully-observed
  data; we refer to this algorithm as "1bitMC" (with some variants on what gets
  capitalized and whether the "1" at the front gets replaced by "one" due to
  Python not allowing variable names to start with numbers):
  - note that at this point, we have both a fast version solved via
    reformulating the problem as a nonconvex optimization problem (and then
    using AdaGrad (Duchi et al 2011)), as well as a slow reference version
    based on using nuclear norm and entrywise max norm projections
    (specifically, we use the three-operator splitting approach by Davis and
    Yin (2017))
- RowCF-kNN (nearest-neighbor-based collaborative filtering):
  - there is both a fast version (using hnswlib (Malkov and Yashunin 2018) for
    approximate nearest neighbor search) and a slow exact version
- a modified version of 1-bit matrix completion meant for handling propensity
  scores that are exactly 1 (we call this algorithm 1bitMC-mod); this is
  discussed in the longer version of the paper where we use it in conjunction
  with a modified logistic regression link function
  - for now there is just a slow reference version (coded in the same manner
    as the 1bitMC slow reference version, i.e., using the Davis-Yin three-way
    operator split)

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

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)

This code uses the following library (on top of Anaconda Python 3.7):
- copt: pip install -U copt

References:

    T. Tony Cai and Wen-Xin Zhou. Matrix completion via max-norm constrained
    optimization. Electronic Journal of Statistics, 10(1):1493–1525, 2016.

    Mark A. Davenport, Yaniv Plan, Ewout Van Den Berg, and Mary Wootters.
    1-bit matrix completion. Information and Inference, 3(3):189–223, 2014

    Damek Davis and Wotao Yin. A three-operator splitting scheme and its
    optimization applications. Set-Valued and Variational Analysis,
    25(4):829-858, 2017.

    John Duchi, Elad Hazan, and Yoram Singer. Adaptive subgradient methods for
    online learning and stochastic optimization. Journal of Machine Learning
    Research, 12(61):2121-2159, 2011.

    Jason D. Lee, Ben Recht, Nathan Srebro, Joel Tropp, and Ruslan R.
    Salakhutdinov. Practical large-scale optimization for max-norm
    regularization. In Advances in Neural Information Processing Systems,
    pages 1297-1305, 2010.

    Yu A. Malkov and Dmitry A. Yashunin. Efficient and robust approximate
    nearest neighbor search using hierarchical navigable small world graphs.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    42(4):824-836, 2018.

    Nathan Srebro and Ruslan R. Salakhutdinov. Collaborative filtering in a
    non-uniform world: Learning with the weighted trace norm. In Advances in
    Neural Information Processing Systems, pages 2056–2064, 2010.
"""
import hnswlib
import numpy as np
import os
import time
from copt import minimize_proximal_gradient, minimize_three_split
from scipy import optimize
from scipy.linalg import svd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.spatial.distance import squareform, pdist
from sklearn.utils.extmath import randomized_svd
from subprocess import DEVNULL, call

from nuc_shrinkage_helper import shrinkage_singular_values
from one_bit_MC_fully_observed_approx_helper import \
    one_bit_MC_fully_observed_approx_helper_adagrad
from weighted_max_norm_helper import weighted_max_norm_helper_adagrad
from doubly_weighted_trace_norm_helper import \
    doubly_weighted_trace_norm_helper_adagrad


def std_logistic_function(x):
    return 1 / (1 + np.exp(-x))


def grad_std_logistic_function(x):
    z = np.exp(-x)
    return z / (1 + z)**2


def row_cf(M, k, thres, metric='hamming'):
    m, n = M.shape
    distances = squareform(pdist(M, metric=metric))
    P_hat = np.zeros((m, n))
    for i in range(m):
        small_k_row_idx = np.argpartition(distances[i], k)[:k]
        P_hat[i] = M[small_k_row_idx, :].mean(axis=0)
    thres_mask = (P_hat < thres)
    if np.any(thres_mask):
        P_hat[thres_mask] = thres
    return P_hat


def row_cf_approx(M, k, thres, hnsw_M=48, hnsw_seed=0, hnsw_search_n_jobs=1):
    m, n = M.shape

    # cosine distance becomes equivalent to Hamming up to scale factor of 2
    M_rescaled = 2*M.astype(np.float32) - 1
    index = hnswlib.Index('cosine', n)
    index.init_index(m, ef_construction=k, M=hnsw_M, random_seed=hnsw_seed)
    # for deterministic index construction, need to use 1 thread
    index.add_items(M_rescaled, num_threads=1)
    index.set_ef(k)
    nearest_neighbor_indices = \
        index.knn_query(M_rescaled, k=k,
                        num_threads=hnsw_search_n_jobs)[0]

    P_hat = np.zeros((m, n))
    for i in range(m):
        P_hat[i] = M[nearest_neighbor_indices[i]].mean(axis=0)
    thres_mask = (P_hat < thres)
    if np.any(thres_mask):
        P_hat[thres_mask] = thres
    return P_hat


def mod_logistic_function(x, gamma, one_minus_logistic_gamma):
    x = np.clip(x, -gamma, gamma)
    return 1 / (1 + np.exp(-x)) + .5 * (1 + x/gamma) * one_minus_logistic_gamma


def grad_mod_logistic_function(x, gamma, one_minus_logistic_gamma):
    x = np.clip(x, -gamma, gamma)
    z = np.exp(-x)
    return z / (1 + z)**2 + one_minus_logistic_gamma / (2*gamma)


def one_bit_MC_fully_observed(M, link, link_gradient, tau, gamma,
                              max_rank=None, opt_max_iter=2000,
                              opt_eps=1e-6):
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
                S = shrinkage_singular_values(S.astype(np.float32),
                                              tau_sqrt_mn)
                _A = np.dot(U * S, VT)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float32),
                                              tau_sqrt_mn)
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


def one_bit_MC_fully_observed_approx(M, rank, gamma, lr=0.1, max_n_epochs=100,
                                     patience=10, loss_tol=1e-4, init_std=0.01,
                                     random_state=None, verbose=0):
    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    m, n = M.shape
    if m > n:
        transpose = True
        M = M.T
        m, n = n, m
    else:
        transpose = False
    M = M.astype(np.uint8)

    U = init_std * rng.randn(m, rank).astype(np.float32)
    V = init_std * rng.randn(n, rank).astype(np.float32)

    # scale U and V to ensure that the entrywise max norm constraint holds
    UV_entrywise_max_norm = \
        np.max([np.abs(np.dot(U, V[col])).max() for col in range(n)])
    if UV_entrywise_max_norm > gamma:
        fix_ratio = np.sqrt(gamma / UV_entrywise_max_norm)
        U *= fix_ratio
        V *= fix_ratio

    U, V = \
        one_bit_MC_fully_observed_approx_helper_adagrad(M, U, V,
                                                        rank, gamma, lr,
                                                        max_n_epochs,
                                                        patience, loss_tol,
                                                        1*verbose, 0)

    if transpose:
        U, V = V, U

    return std_logistic_function(np.dot(U, V.T))


def one_bit_MC_mod_fully_observed(M, link, link_gradient, tau, gamma,
                                  max_rank=None, opt_max_iter=2000,
                                  opt_eps=1e-6, phi=None):
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
                S = shrinkage_singular_values(S.astype(np.float32),
                                              tau_sqrt_mn)
                _A = np.dot(U * S, VT)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
            if S.sum() > tau_sqrt_mn:
                S = shrinkage_singular_values(S.astype(np.float32),
                                              tau_sqrt_mn)
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


def approx_doubly_weighted_trace_norm_unconstrained(X, M, W, n_components,
                                                    lmbda, init_std=0.01,
                                                    opt_max_iter=100,
                                                    opt_eps=1e-6,
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

    sqrt_row_weights = np.sqrt(row_weights)
    sqrt_col_weights = np.sqrt(col_weights)

    U = init_std * rng.randn(m, n_components).astype(np.float32)
    V = init_std * rng.randn(n, n_components).astype(np.float32)
    m_times_n_components = m * n_components

    nnz = M.sum()

    def f(_UV, return_gradient=True):
        U = _UV[:m_times_n_components].reshape(m, n_components)
        V = _UV[m_times_n_components:].reshape(n, n_components)
        X_hat_tmp = U.dot(V.T)
        diff = (X_hat_tmp - X) * M * np.sqrt(W)

        U_weighted = U * sqrt_row_weights[:, np.newaxis]
        V_weighted = V * sqrt_col_weights[:, np.newaxis]

        # by default, np.linalg.norm for matrices is Frobenius norm
        loss = (np.linalg.norm(diff)**2
                + lmbda * (np.linalg.norm(U_weighted)**2
                           + np.linalg.norm(V_weighted)**2)) / 2 / nnz

        if not return_gradient:
            return loss

        U_grad = np.dot(diff, V) + lmbda * U_weighted
        V_grad = np.dot(diff.T, U) + lmbda * V_weighted
        UV_grad = np.concatenate([U_grad.flatten(), V_grad.flatten()])
        UV_grad /= nnz
        return loss, UV_grad

    _UV_hat = \
        minimize(f, np.concatenate([U.flatten(), V.flatten()]),
                 method='L-BFGS-B', jac=True,
                 options={'maxiter': opt_max_iter},
                 tol=opt_eps).x
    U = _UV_hat[:m_times_n_components].reshape(m, n_components)
    V = _UV_hat[m_times_n_components:].reshape(n, n_components)
    return U.dot(V.T)


def approx_doubly_weighted_trace_norm(X, M, W, n_components,
                                      lmbda, alpha, init_std=0.01,
                                      max_n_epochs=500, lr=0.1, patience=10,
                                      loss_tol=1e-6, random_state=None,
                                      trace_norm_weighting=True):
    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    M_coo = coo_matrix(M)
    X_rows = M_coo.row.astype(np.int64)
    X_cols = M_coo.col.astype(np.int64)
    X_values = np.array([X[row, col] for row, col in zip(X_rows, X_cols)],
                        dtype=np.float32)
    weights = np.array([W[row, col] for row, col in zip(X_rows, X_cols)],
                       dtype=np.float32)

    m, n = X.shape

    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    U = init_std * rng.randn(m, n_components).astype(np.float32)
    V = init_std * rng.randn(n, n_components).astype(np.float32)

    # scale U and V to ensure that the entrywise max norm constraint holds
    UV_entrywise_max_norm = \
        np.max([np.abs(np.dot(U, V[col])).max() for col in range(n)])
    if UV_entrywise_max_norm > alpha:
        fix_ratio = np.sqrt(alpha / UV_entrywise_max_norm)
        U *= fix_ratio
        V *= fix_ratio

    if trace_norm_weighting:
        total = float(np.sum(X))
        row_weights = np.sum(X, axis=1) / total * m
        col_weights = np.sum(X, axis=0) / total * n
    else:
        row_weights = np.ones(m)
        col_weights = np.ones(n)

    sqrt_row_weights = np.sqrt(row_weights).astype(np.float32)
    sqrt_col_weights = np.sqrt(col_weights).astype(np.float32)

    U, V = doubly_weighted_trace_norm_helper_adagrad(
        X_rows, X_cols, X_values, weights, sqrt_row_weights, sqrt_col_weights,
        U, V, lmbda, alpha, lr, max_n_epochs, patience, loss_tol, 0)
    return np.dot(U, V.T)


def weighted_max_norm(X, M, W, n_components, R, alpha, lr=0.1,
                      max_n_epochs=500, patience=10, loss_tol=1e-6,
                      init_std=0.01, random_state=None, verbose=0):
    M_coo = coo_matrix(M)
    X_rows = M_coo.row.astype(np.int64)
    X_cols = M_coo.col.astype(np.int64)
    X_values = np.array([X[row, col] for row, col in zip(X_rows, X_cols)],
                        dtype=np.float32)
    weights = np.array([W[row, col] for row, col in zip(X_rows, X_cols)],
                       dtype=np.float32)

    m, n = X.shape
    sqrt_R = np.sqrt(R)

    if random_state is None:
        rng = np.random.RandomState()
    elif type(random_state) == np.random.RandomState:
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    U = init_std * rng.randn(m, n_components).astype(np.float32)
    V = init_std * rng.randn(n, n_components).astype(np.float32)

    # scale U and V to ensure that the entrywise max norm constraint holds
    UV_entrywise_max_norm = \
        np.max([np.abs(np.dot(U, V[col])).max() for col in range(n)])
    if UV_entrywise_max_norm > alpha:
        fix_ratio = np.sqrt(alpha / UV_entrywise_max_norm)
        U *= fix_ratio
        V *= fix_ratio
    U = max_row_l2_norm_proj(U, sqrt_R)
    V = max_row_l2_norm_proj(V, sqrt_R)

    U, V = weighted_max_norm_helper_adagrad(X_rows, X_cols, X_values, weights,
                                            U, V, sqrt_R, alpha, lr,
                                            max_n_epochs, patience, loss_tol,
                                            verbose)
    return np.dot(U, V.T)


def max_row_l2_norm_proj(U, threshold):
    for u, row in enumerate(U):
        row_l2_norm = np.linalg.norm(row)
        if row_l2_norm > threshold:
            U[u] = row / row_l2_norm * threshold
    return U
