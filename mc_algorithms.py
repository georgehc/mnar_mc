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
- apgpy: https://github.com/bodono/apgpy

References:

    T. Tony Cai and Wen-Xin Zhou. Matrix completion via max-norm constrained
    optimization.Elec-tronic Journal of Statistics, 10(1):1493–1525, 2016.

    Mark A. Davenport, Yaniv Plan, Ewout Van Den Berg, and Mary Wootters.
    1-bit matrix completion.Information and Inference, 3(3):189–223, 2014

    Nathan Srebro and Ruslan R. Salakhutdinov. Collaborative filtering in a
    non-uniform world: Learning with the weighted trace norm. In Advances in
    Neural Information Processing Systems, pages 2056–2064, 2010.
"""
import numpy as np
import apgpy
from sklearn.utils.extmath import randomized_svd


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


def one_bit_MC_fully_observed(M, link, link_gradient, tau, gamma, max_rank=None,
                              apg_max_iter=100, apg_eps=1e-12,
                              apg_use_restart=True):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m = M.shape[0]
    n = M.shape[1]
    tau_sqrt_mn = tau * np.sqrt(m*n)

    def prox(_A, t):
        _A = _A.reshape(m, n)

        # project so nuclear norm is at most tau*sqrt(m*n)
        if max_rank is None:
            U, S, VT = np.linalg.svd(_A, full_matrices=False)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
        nuclear_norm = np.sum(S)
        if nuclear_norm > tau_sqrt_mn:
            S *= tau_sqrt_mn / nuclear_norm
            _A = np.dot(U * S, VT)

        # clip matrix entries with absolute value greater than gamma
        mask = np.abs(_A) > gamma
        if mask.sum() > 0:
            _A[mask] = np.sign(_A[mask]) * gamma

        return _A.flatten()

    M_one_mask = (M == 1)
    M_zero_mask = (M == 0)
    def grad(_A):
        _A = _A.reshape(m, n)

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask])/link(_A[M_one_mask])
        grad[M_zero_mask] = \
            link_gradient(_A[M_zero_mask])/(1 - link(_A[M_zero_mask]))

        return grad.flatten()

    A_hat = apgpy.solve(grad, prox, np.zeros(m*n),
                        max_iters=apg_max_iter,
                        eps=apg_eps,
                        use_gra=True,
                        use_restart=apg_use_restart,
                        quiet=True)
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def one_bit_MC_mod_fully_observed(M, link, link_gradient, tau, gamma,
                                  max_rank=None, apg_max_iter=100,
                                  apg_eps=1e-12, apg_use_restart=True,
                                  phi=None):
    # parameters are the same as in the paper; if `max_rank` is set to None,
    # then exact SVD is used
    m = M.shape[0]
    n = M.shape[1]
    tau_sqrt_mn = tau * np.sqrt(m*n)
    M_zero_mask = (M == 0)
    if phi is None:
        phi = .95 * gamma

    def prox(_A, t):
        _A = _A.reshape(m, n)

        # project so nuclear norm is at most tau*sqrt(m*n)
        if max_rank is None:
            U, S, VT = np.linalg.svd(_A, full_matrices=False)
        else:
            U, S, VT = randomized_svd(_A, max_rank)
        nuclear_norm = np.sum(S)
        if nuclear_norm > tau_sqrt_mn:
            S *= tau_sqrt_mn / nuclear_norm
            _A = np.dot(U * S, VT)

        # clip matrix entries with absolute value greater than gamma
        mask = np.abs(_A) > gamma
        if mask.sum() > 0:
            _A[mask] = np.sign(_A[mask]) * gamma

        mask = _A[M_zero_mask] > phi
        if mask.sum() > 0:
            _A[M_zero_mask][mask] = phi

        return _A.flatten()

    M_one_mask = (M == 1)
    def grad(_A):
        _A = _A.reshape(m, n)

        grad = np.zeros((m, n))
        grad[M_one_mask] = -link_gradient(_A[M_one_mask])/ \
            link(np.maximum(_A[M_one_mask], -gamma))
        grad[M_zero_mask] = \
            link_gradient(_A[M_zero_mask])/ \
            (1 - link(np.minimum(_A[M_zero_mask], phi)))

        return grad.flatten()

    A_hat = apgpy.solve(grad, prox, np.zeros(m*n),
                        max_iters=apg_max_iter,
                        eps=apg_eps,
                        use_gra=True,
                        use_restart=apg_use_restart,
                        quiet=True)
    P_hat = link(A_hat.reshape(m, n))
    return P_hat


def weighted_softimpute(X, M, W, lmbda, max_rank=None,
                        min_value=None, max_value=None,
                        apg_max_iter=100, apg_eps=1e-6,
                        apg_use_restart=True):
    # if `max_rank` is set to None, then exact SVD is used
    m = X.shape[0]
    n = X.shape[1]

    def prox(Z, t):
        Z = Z.reshape(m, n)

        # singular value shrinkage
        if max_rank is None:
            U, S, VT = np.linalg.svd(Z, full_matrices=False)
        else:
            U, S, VT = randomized_svd(Z, max_rank)
        S = np.maximum(S - lmbda*t, 0)
        Z = np.dot(U * S, VT)

        # clip values
        if min_value is not None:
            mask = Z < min_value
            if mask.sum() > 0:
                Z[mask] = min_value
        if max_value is not None:
            mask = Z > max_value
            if mask.sum() > 0:
                Z[mask] = max_value

        return Z.flatten()

    M_one_mask = (M == 1)
    masked_weights = W[M_one_mask]
    masked_X = X[M_one_mask]
    def grad(Z):
        grad = np.zeros((m, n))
        grad[M_one_mask] = (Z.reshape(m, n)[M_one_mask] - masked_X) * masked_weights
        return grad.flatten()

    X_hat = apgpy.solve(grad, prox, np.zeros(m*n),
                        max_iters=apg_max_iter,
                        eps=apg_eps,
                        use_gra=True,
                        use_restart=apg_use_restart,
                        quiet=True).reshape((m, n))
    return X_hat


class DoublyWeightedTraceNorm():
    def __init__(self, X, k, alpha, lambda1, max_iter=100,
                 tol=1e-6, apg_use_restart=True, verbose=False,
                 X_vad=None, random_state=None):
        self.m, self.n = X.shape
        self.X = X
        self.data_row, self.data_col = np.nonzero(X)
        self.data = X[self.data_row, self.data_col]
        self.X_O = X.copy()
        self.X_O.data = np.ones_like(self.X_O.data)
        self.k = k
        self.alpha = alpha
        self.lambda1 = lambda1
        self.max_iter = max_iter
        self.tol = tol
        self.X_vad = X_vad
        self.apg_use_restart = apg_use_restart
        self.verbose = verbose

        if X_vad is not None:
            self.data_row_vad, self.data_col_vad = np.nonzero(X_vad)
            self.data_vad = X_vad.data

        if random_state is None:
            self.rng = np.random.RandomState()
        elif type(random_state) == np.random.RandomState:
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

    def compute_rc_weight(self):
        tot = np.float(np.sum(self.X_O))
        col_count = np.array(np.sum(self.X_O, axis=0)).flatten()
        row_count = np.array(np.sum(self.X_O, axis=1)).flatten()
        col_p = np.array(np.sum(self.X_O, axis=0)).flatten() / tot
        row_p = np.array(np.sum(self.X_O, axis=1)).flatten() / tot
        return tot, row_count, col_count, row_p, col_p

    def init_weights(self):
        self.U = self.rng.rand(self.m, self.k).astype(np.float)
        self.V = self.rng.rand(self.n, self.k).astype(np.float)

    def fit(self, P=None):
        n_users, n_items = self.X.shape
        self.X_train_mask = (self.X > 0).astype(np.float)
        self.init_weights()
        if P is None:
            self._update()
        else:
            self._update(1.0 / P)
        return self

    def _update(self, W=None):
        X = self.X
        m, n = X.shape
        if W is None:
            W = np.ones((m, n))
        tot, rn, cn, rp, cp = self.compute_rc_weight()

        def grad(_UV):
            assert (len(_UV) == m * self.k + n * self.k)
            U = _UV[:m * self.k].reshape((m, self.k))
            V = _UV[m * self.k:].reshape((n, self.k))
            X_hat_tmp = U.dot(V.T)
            grad_U = -np.dot((X - X_hat_tmp) * W * self.X_train_mask, V)
            grad_V = -np.dot(U.T, (X - X_hat_tmp) * W * self.X_train_mask).T
            grad_UV = np.concatenate([grad_U.flatten(), grad_V.flatten()])
            return grad_UV

        def l2_prox(x, t):
            return np.maximum(
                1 - t / np.maximum(np.linalg.norm(x, 2), 1e-6), 0.0) * x

        def prox(_UV, t):
            assert (len(_UV) == m * self.k + n * self.k)
            U_tmp = _UV[:m * self.k].reshape((m, self.k))
            V_tmp = _UV[m * self.k:].reshape((n, self.k))
            U = U_tmp.copy()
            for i in range(m):
                U[i] = l2_prox(U_tmp[i], self.lambda1 /
                               2 *
                               (np.power(rp[i], self.alpha) /
                                np.maximum(rn[i], 1)) *
                               t)
            V = V_tmp.copy()
            for i in range(n):
                V[i] = l2_prox(V_tmp[i], self.lambda1 /
                               2 *
                               (np.power(cp[i], self.alpha) /
                                np.maximum(cn[i], 1)) *
                               t)
            prox_UV = np.concatenate([U.flatten(), V.flatten()])
            return prox_UV

        _UV_hat = apgpy.solve(grad,
                              prox,
                              np.concatenate([self.U.flatten(),
                                              self.V.flatten()]),
                              max_iters=self.max_iter,
                              eps=self.tol,
                              use_restart=self.apg_use_restart,
                              quiet=(not self.verbose),
                              use_gra=True)
        U = _UV_hat[:m * self.k].reshape((m, self.k))
        V = _UV_hat[m * self.k:].reshape((n, self.k))
        self.U = U
        self.V = V
        self.X_hat = self.U.dot(self.V.T)
        return self


class MaxNorm():
    def __init__(self, n_components=20, tol=1e-6, max_iter=100, init_std=0.01,
                 R=0.1, alpha=5, verbose=False, apg_use_restart=True,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.init_std = init_std
        self.R = R
        self.alpha = alpha
        self.verbose = verbose
        self.tol = tol
        self.X_hat = None
        self.apg_use_restart = apg_use_restart

        if random_state is None:
            self.rng = np.random.RandomState()
        elif type(random_state) == np.random.RandomState:
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        self.U = self.init_std * \
            self.rng.rand(n_users, self.n_components).astype(np.float)
        self.V = self.init_std * \
            self.rng.rand(n_items, self.n_components).astype(np.float)

    def fit(self, X, **kwargs):
        n_users, n_items = X.shape
        self.X_train_mask = (X > 0).astype(np.float)
        self._init_params(n_users, n_items)
        self._update(X)
        return self

    def _update(self, X, **kwargs):
        m, n = X.shape

        def grad(_UV):
            assert (len(_UV) == m * self.n_components + n * self.n_components)
            U = _UV[:m * self.n_components].reshape((m, self.n_components))
            V = _UV[m * self.n_components:].reshape((n, self.n_components))
            X_hat_tmp = U.dot(V.T)
            grad_U = -np.dot((X - X_hat_tmp) * self.X_train_mask, V)
            grad_V = -np.dot(U.T, (X - X_hat_tmp) * self.X_train_mask).T
            grad_UV = np.concatenate([grad_U.flatten(), grad_V.flatten()])
            return grad_UV

        def prox(_UV, t):
            assert (len(_UV) == m * self.n_components + n * self.n_components)
            U_tmp = _UV[:m * self.n_components].reshape((m, self.n_components))
            V_tmp = _UV[m * self.n_components:].reshape((n, self.n_components))
            tmp_norm_inf = np.linalg.norm(U_tmp.dot(V_tmp.T), np.inf)
            if tmp_norm_inf > self.alpha:
                U_tmp = np.sqrt(self.alpha) / np.sqrt(tmp_norm_inf) * U_tmp
                V_tmp = np.sqrt(self.alpha) / np.sqrt(tmp_norm_inf) * V_tmp
            U = self._proj(U_tmp, self.R)
            V = self._proj(V_tmp, self.R)
            prox_UV = np.concatenate([U.flatten(), V.flatten()])
            return prox_UV

        _UV_hat = apgpy.solve(grad,
                              prox,
                              np.concatenate([self.U.flatten(),
                                              self.V.flatten()]),
                              max_iters=self.max_iter,
                              eps=self.tol,
                              use_restart=self.apg_use_restart,
                              quiet=(not self.verbose),
                              use_gra=True)
        U = _UV_hat[:m * self.n_components].reshape((m, self.n_components))
        V = _UV_hat[m * self.n_components:].reshape((n, self.n_components))
        self.U = U
        self.V = V
        self.X_hat = self.U.dot(self.V.T)
        return self

    def _proj(self, U, R):
        n, k = U.shape
        for i in range(n):
            # print (np.linalg.norm(U[i], 2))
            if np.linalg.norm(U[i], 2) > R:
                U[i] = U[i] / np.linalg.norm(U[i], 2) * R
                # print (np.linalg.norm(U[i], 2))
        return U


class WeightedMaxNorm():
    def __init__(self, n_components=20, tol=1e-6, max_iter=100, init_std=0.01,
                 R=0.1, alpha=5, verbose=False, apg_use_restart=True,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.init_std = init_std
        self.R = R
        self.alpha = alpha
        self.verbose = verbose
        self.tol = tol
        self.X_hat = None
        self.apg_use_restart = apg_use_restart

        if random_state is None:
            self.rng = np.random.RandomState()
        elif type(random_state) == np.random.RandomState:
            self.rng = random_state
        else:
            self.rng = np.random.RandomState(random_state)

    def _init_params(self, n_users, n_items):
        ''' Initialize all the latent factors '''
        self.U = self.init_std * \
            self.rng.rand(n_users, self.n_components).astype(np.float)
        self.V = self.init_std * \
            self.rng.rand(n_items, self.n_components).astype(np.float)

    def fit(self, X, P, **kwargs):
        n_users, n_items = X.shape
        self.X_train_mask = (X > 0).astype(np.float)
        self._init_params(n_users, n_items)
        self._update(X, 1.0 / P)
        return self

    def _update(self, X, W, **kwargs):
        m, n = X.shape

        def grad(_UV):
            assert (len(_UV) == m * self.n_components + n * self.n_components)
            U = _UV[:m * self.n_components].reshape((m, self.n_components))
            V = _UV[m * self.n_components:].reshape((n, self.n_components))
            X_hat_tmp = U.dot(V.T)
            grad_U = -np.dot((X - X_hat_tmp) * W * self.X_train_mask, V)
            grad_V = -np.dot(U.T, (X - X_hat_tmp) * W * self.X_train_mask).T
            grad_UV = np.concatenate([grad_U.flatten(), grad_V.flatten()])
            return grad_UV

        def prox(_UV, t):
            assert (len(_UV) == m * self.n_components + n * self.n_components)
            U_tmp = _UV[:m * self.n_components].reshape((m, self.n_components))
            V_tmp = _UV[m * self.n_components:].reshape((n, self.n_components))
            tmp_norm_inf = np.linalg.norm(U_tmp.dot(V_tmp.T), np.inf)
            if tmp_norm_inf > self.alpha:
                U_tmp = np.sqrt(self.alpha) / np.sqrt(tmp_norm_inf) * U_tmp
                V_tmp = np.sqrt(self.alpha) / np.sqrt(tmp_norm_inf) * V_tmp
            U = self._proj(U_tmp, self.R)
            V = self._proj(V_tmp, self.R)
            prox_UV = np.concatenate([U.flatten(), V.flatten()])
            return prox_UV

        _UV_hat = apgpy.solve(grad,
                              prox,
                              np.concatenate([self.U.flatten(),
                                              self.V.flatten()]),
                              max_iters=self.max_iter,
                              eps=self.tol,
                              use_restart=self.apg_use_restart,
                              quiet=(not self.verbose))
        U = _UV_hat[:m * self.n_components].reshape((m, self.n_components))
        V = _UV_hat[m * self.n_components:].reshape((n, self.n_components))
        self.U = U
        self.V = V
        self.X_hat = self.U.dot(self.V.T)
        return self

    def _proj(self, U, R):
        n, k = U.shape
        for i in range(n):
            if np.linalg.norm(U[i], 2) > R:
                U[i] = U[i] / np.linalg.norm(U[i], 2) * R
        return U
