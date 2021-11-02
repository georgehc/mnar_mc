"""
Wrappers for some matrix completion algorithms to make them work with Surprise

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
"""
import json
import hashlib
import numpy as np
import os
import time
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegressionCV
from soft_impute_ALS import WeightedSoftImpute_ALS
from surprise import AlgoBase, PredictionImpossible
from subprocess import DEVNULL, call
from mc_algorithms import one_bit_MC_fully_observed, \
        one_bit_MC_fully_observed_approx, std_logistic_function, \
        grad_std_logistic_function, weighted_softimpute, \
        one_bit_MC_mod_fully_observed, mod_logistic_function, \
        grad_mod_logistic_function, approx_doubly_weighted_trace_norm, \
        approx_doubly_weighted_trace_norm_unconstrained, weighted_max_norm, \
        row_cf, row_cf_approx
from expomf import ExpoMF


cache_dir = 'cache_propensity_estimates'


class WeightedSoftImputeWrapper(AlgoBase):
    def __init__(self, max_rank, lmbda, min_value=None, max_value=None,
                 max_iter=200, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., one_bit_mc_approx_lr=.1,
                 row_cf_k=10, row_cf_thres=0.1, approx=True,
                 user_features=None, item_features=None, min_prob=.05,
                 mcar_data=None, verbose=False, cache_prefix=''):
        self.max_rank = max_rank
        self.lmbda = lmbda
        self.min_value = min_value
        self.max_value = max_value
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.one_bit_mc_approx_lr = one_bit_mc_approx_lr
        self.row_cf_k = row_cf_k
        self.row_cf_thres = row_cf_thres
        self.approx = approx
        self.user_features = user_features
        self.item_features = item_features
        self.min_prob = min_prob
        self.mcar_data = mcar_data
        self.verbose = verbose
        self.cache_prefix = cache_prefix

        AlgoBase.__init__(self)

    def fit(self, trainset):
        linux_use_all_cores()

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if self.propensity_scores == 'naive bayes':
            P_est_input = X
        else:
            P_est_input = M
        P_hat = compute_propensity_scores(P_est_input,
                                          self.propensity_scores,
                                          (trainset.n_users,
                                           trainset.n_items),
                                          self.one_bit_mc_tau,
                                          self.one_bit_mc_gamma,
                                          self.one_bit_mc_max_rank,
                                          self.one_bit_mc_approx_lr,
                                          self.row_cf_k,
                                          self.row_cf_thres,
                                          self.approx,
                                          self.verbose,
                                          self.user_features,
                                          self.item_features,
                                          self.min_prob,
                                          self.mcar_data,
                                          self.cache_prefix)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        self.predictions = weighted_softimpute(X, M, W, self.lmbda,
                                               min_value=self.min_value,
                                               max_value=self.max_value,
                                               max_rank=self.max_rank,
                                               opt_max_iter=self.max_iter)

        # modify training entries to be their observed values
        for u, i, r in trainset.all_ratings():
            self.predictions[u, i] = r

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


class WeightedSoftImputeALSWrapper(AlgoBase):
    def __init__(self, max_rank, lmbda, min_value=None, max_value=None,
                 max_iter=200, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., one_bit_mc_approx_lr=.1,
                 row_cf_k=10, row_cf_thres=0.1, approx=True,
                 user_features=None, item_features=None, min_prob=.05,
                 mcar_data=None, verbose=False, cache_prefix=''):
        self.max_rank = max_rank
        self.lmbda = lmbda
        self.min_value = min_value
        self.max_value = max_value
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.one_bit_mc_approx_lr = one_bit_mc_approx_lr
        self.row_cf_k = row_cf_k
        self.row_cf_thres = row_cf_thres
        self.approx = approx
        self.user_features = user_features
        self.item_features = item_features
        self.min_prob = min_prob
        self.mcar_data = mcar_data
        self.verbose = verbose
        self.cache_prefix = cache_prefix

        AlgoBase.__init__(self)

    def fit(self, trainset):
        linux_use_all_cores()

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if self.propensity_scores == 'naive bayes':
            P_est_input = X
        else:
            P_est_input = M
        P_hat = compute_propensity_scores(P_est_input,
                                          self.propensity_scores,
                                          (trainset.n_users,
                                           trainset.n_items),
                                          self.one_bit_mc_tau,
                                          self.one_bit_mc_gamma,
                                          self.one_bit_mc_max_rank,
                                          self.one_bit_mc_approx_lr,
                                          self.row_cf_k,
                                          self.row_cf_thres,
                                          self.approx,
                                          self.verbose,
                                          self.user_features,
                                          self.item_features,
                                          self.min_prob,
                                          self.mcar_data,
                                          self.cache_prefix)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        sparse_row_indices, sparse_col_indices, sparse_data = \
            zip(*[(u, i, r) for u, i, r in trainset.all_ratings()])
        X_incomplete_csr = \
            csr_matrix((sparse_data, (sparse_row_indices, sparse_col_indices)),
                       shape=(trainset.n_users, trainset.n_items)).tocsr()

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        sals = WeightedSoftImpute_ALS(self.max_rank, X_incomplete_csr, W,
                                      self.verbose)
        sals.fit(Lambda=self.lmbda, maxit=self.max_iter)
        X_hat = sals._U.dot(sals._Dsq.dot(sals._V.T))

        # only clip afterward (otherwise the optimization gets more
        # complicated)
        if self.min_value is not None or self.max_value is not None:
            X_hat = np.clip(X_hat, self.min_value, self.max_value)
        self.predictions = X_hat

        # modify training entries to be their observed values
        for u, i, r in trainset.all_ratings():
            self.predictions[u, i] = r

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


class ExpoMFWrapper(AlgoBase):
    def __init__(self, min_value=None, max_value=None, n_components=20,
                 max_iter=100, batch_size=1024, init_std=0.01,
                 random_state=None):
        self.min_value = min_value
        self.max_value = max_value
        AlgoBase.__init__(self)
        self.mc = ExpoMF(n_components=n_components, max_iter=max_iter,
                         init_std=init_std, random_state=random_state)

    def fit(self, trainset):
        linux_use_all_cores()

        AlgoBase.fit(self, trainset)
        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r
        Y = csr_matrix(Y)
        self.mc.fit(Y)
        U, V = self.mc.theta, self.mc.beta
        X_hat = U.dot(V.T)

        # only clip afterward since otherwise for correctness we would have to
        # carefully change the optimization...
        if self.min_value is not None or self.max_value is not None:
            X_hat = np.clip(X_hat, self.min_value, self.max_value)
        self.predictions = X_hat

        # modify training entries to be their observed values
        for u, i, r in trainset.all_ratings():
            self.predictions[u, i] = r

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


class DoublyWeightedTraceNormWrapper(AlgoBase):
    def __init__(self, min_value=None, max_value=None, n_components=3,
                 lmbda=10, init_std=0.01, max_iter=30, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., one_bit_mc_approx_lr=.1,
                 row_cf_k=10, row_cf_thres=0.1, approx=True,
                 random_state=None, trace_norm_weighting=True,
                 user_features=None, item_features=None, min_prob=.05,
                 mcar_data=None, verbose=False, cache_prefix=''):
        self.min_value = min_value
        self.max_value = max_value
        self.n_components = n_components
        self.lmbda = lmbda
        self.init_std = init_std
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.one_bit_mc_approx_lr = one_bit_mc_approx_lr
        self.random_state = random_state
        self.row_cf_k = row_cf_k
        self.row_cf_thres = row_cf_thres
        self.approx = approx
        self.user_features = user_features
        self.item_features = item_features
        self.min_prob = min_prob
        self.mcar_data = mcar_data
        self.verbose = verbose
        self.trace_norm_weighting = trace_norm_weighting
        self.cache_prefix = cache_prefix

        AlgoBase.__init__(self)

    def fit(self, trainset):
        linux_use_all_cores()

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if self.propensity_scores == 'naive bayes':
            P_est_input = X
        else:
            P_est_input = M
        P_hat = compute_propensity_scores(P_est_input,
                                          self.propensity_scores,
                                          (trainset.n_users,
                                           trainset.n_items),
                                          self.one_bit_mc_tau,
                                          self.one_bit_mc_gamma,
                                          self.one_bit_mc_max_rank,
                                          self.one_bit_mc_approx_lr,
                                          self.row_cf_k,
                                          self.row_cf_thres,
                                          self.approx,
                                          self.verbose,
                                          self.user_features,
                                          self.item_features,
                                          self.min_prob,
                                          self.mcar_data,
                                          self.cache_prefix)
        self.propensity_estimates = P_hat

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        if self.min_value is None and self.max_value is None:
            self.predictions = \
                approx_doubly_weighted_trace_norm_unconstrained(
                    X, M, W, self.n_components, self.lmbda,
                    init_std=self.init_std, opt_max_iter=self.max_iter,
                    random_state=self.random_state,
                    trace_norm_weighting=self.trace_norm_weighting)
        else:
            alpha = max(abs(self.min_value), abs(self.max_value))
            X_hat = \
                approx_doubly_weighted_trace_norm(
                    X, M, W, self.n_components, self.lmbda, alpha,
                    init_std=self.init_std, max_n_epochs=self.max_iter,
                    random_state=self.random_state,
                    trace_norm_weighting=self.trace_norm_weighting)
            self.predictions = np.clip(X_hat, self.min_value, self.max_value)

        # modify training entries to be their observed values
        for u, i, r in trainset.all_ratings():
            self.predictions[u, i] = r

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


class WeightedMaxNormWrapper(AlgoBase):
    def __init__(self, min_value=1, max_value=5, n_components=20, R=1,
                 init_std=0.01, max_iter=100, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., one_bit_mc_approx_lr=.1, row_cf_k=10,
                 row_cf_thres=0.1, approx=True, user_features=None,
                 item_features=None, min_prob=.05, verbose=False,
                 random_state=None, mcar_data=None, cache_prefix=''):
        self.min_value = min_value
        self.max_value = max_value
        self.n_components = n_components
        self.R = R
        self.init_std = init_std
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.one_bit_mc_approx_lr = one_bit_mc_approx_lr
        self.row_cf_k = row_cf_k
        self.row_cf_thres = row_cf_thres
        self.approx = approx
        self.user_features = user_features
        self.item_features = item_features
        self.min_prob = min_prob
        self.mcar_data = mcar_data
        self.verbose = verbose
        self.random_state = random_state
        self.cache_prefix = cache_prefix

        AlgoBase.__init__(self)

    def fit(self, trainset):
        linux_use_all_cores()

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if self.propensity_scores == 'naive bayes':
            P_est_input = X
        else:
            P_est_input = M
        P_hat = compute_propensity_scores(P_est_input,
                                          self.propensity_scores,
                                          (trainset.n_users,
                                           trainset.n_items),
                                          self.one_bit_mc_tau,
                                          self.one_bit_mc_gamma,
                                          self.one_bit_mc_max_rank,
                                          self.one_bit_mc_approx_lr,
                                          self.row_cf_k,
                                          self.row_cf_thres,
                                          self.approx,
                                          self.verbose,
                                          self.user_features,
                                          self.item_features,
                                          self.min_prob,
                                          self.mcar_data,
                                          self.cache_prefix)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        alpha = max(abs(self.min_value), abs(self.max_value))
        X_hat = \
            weighted_max_norm(X, M, W, self.n_components,
                              self.R, alpha,
                              max_n_epochs=self.max_iter,
                              init_std=self.init_std,
                              random_state=self.random_state,
                              verbose=1*(self.verbose))
        self.predictions = np.clip(X_hat, self.min_value, self.max_value)

        # modify training entries to be their observed values
        for u, i, r in trainset.all_ratings():
            self.predictions[u, i] = r

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


def compute_and_save_propensity_scores_1bitmc(M, one_bit_mc_tau,
                                              one_bit_mc_gamma,
                                              one_bit_mc_max_rank,
                                              one_bit_mc_approx_lr,
                                              approx, verbose=False,
                                              cache_prefix=''):
    if verbose:
        print('Estimating propensity scores via matrix completion...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        if one_bit_mc_gamma > 0 and approx:
            effective_rank = \
                int(min(np.floor((one_bit_mc_tau / one_bit_mc_gamma)**2),
                        one_bit_mc_max_rank))
            prefix = '1bitmc_approx_g%f_r%d_lr%f_' \
                % (one_bit_mc_gamma, effective_rank, one_bit_mc_approx_lr)
        else:
            prefix = '1bitmc_t%f_g%f_r%d_' \
                % (one_bit_mc_tau, one_bit_mc_gamma, one_bit_mc_max_rank)
        cache_filename = os.path.join(cache_dir,
                                      cache_prefix
                                      + prefix
                                      + memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            os.makedirs(cache_dir, exist_ok=True)
            if one_bit_mc_tau == 0 or one_bit_mc_gamma == 0:
                tic = time.time()
                P_hat = 0.5*np.ones((m, n))
                elapsed = time.time() - tic
            elif approx:
                tic = time.time()
                P_hat = \
                    one_bit_MC_fully_observed_approx(
                        M,
                        effective_rank,
                        one_bit_mc_gamma,
                        lr=one_bit_mc_approx_lr,
                        random_state=0,
                        verbose=1*(verbose))
                elapsed = time.time() - tic
            else:
                tic = time.time()
                P_hat = one_bit_MC_fully_observed(M, std_logistic_function,
                                                  grad_std_logistic_function,
                                                  one_bit_mc_tau,
                                                  one_bit_mc_gamma,
                                                  max_rank=one_bit_mc_max_rank)
                elapsed = time.time() - tic
            try:
                np.savetxt(cache_filename, P_hat)
                np.savetxt(cache_filename[:-4] + '_time.txt',
                           np.array(elapsed).reshape(1, -1))
            except Exception:
                pass
        else:
            try:
                P_hat = np.loadtxt(cache_filename)
                if P_hat.shape[0] != m or P_hat.shape[1] != n:
                    print('*** WARNING: Recomputing propensity scores '
                          + '(mismatched dimensions in cached file)')
                    try:
                        os.remove(cache_filename)
                    except Exception:
                        pass
                    continue
            except ValueError:
                print('*** WARNING: Recomputing propensity scores '
                      + '(malformed numpy array encountered)')
                try:
                    os.remove(cache_filename)
                except Exception:
                    pass
                continue
        break
    return P_hat


def compute_and_save_propensity_scores_1bitmc_mod(M, one_bit_mc_tau,
                                                  one_bit_mc_gamma,
                                                  one_bit_mc_max_rank,
                                                  verbose=False,
                                                  cache_prefix=''):
    if verbose:
        print('Estimating propensity scores via matrix completion...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        prefix = '1bitmc-mod_t%f_g%f_r%d_' \
            % (one_bit_mc_tau, one_bit_mc_gamma, one_bit_mc_max_rank)
        cache_filename = os.path.join(cache_dir,
                                      cache_prefix
                                      + prefix
                                      + memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            tic = time.time()
            one_minus_logistic_gamma \
                = 1 - std_logistic_function(one_bit_mc_gamma)

            def link(x):
                return mod_logistic_function(x, one_bit_mc_gamma,
                                             one_minus_logistic_gamma)

            def grad_link(x):
                return grad_mod_logistic_function(
                    x, one_bit_mc_gamma, one_minus_logistic_gamma)

            os.makedirs(cache_dir, exist_ok=True)
            P_hat = one_bit_MC_mod_fully_observed(M, link, grad_link,
                                                  one_bit_mc_tau,
                                                  one_bit_mc_gamma,
                                                  max_rank=one_bit_mc_max_rank)
            elapsed = time.time() - tic
            try:
                np.savetxt(cache_filename, P_hat)
                np.savetxt(cache_filename[:-4] + '_time.txt',
                           np.array(elapsed).reshape(1, -1))
            except Exception:
                pass
        else:
            try:
                P_hat = np.loadtxt(cache_filename)
                if P_hat.shape[0] != m or P_hat.shape[1] != n:
                    print('*** WARNING: Recomputing propensity scores '
                          + '(mismatched dimensions in cached file)')
                    try:
                        os.remove(cache_filename)
                    except Exception:
                        pass
                    continue
            except ValueError:
                print('*** WARNING: Recomputing propensity scores '
                      + '(malformed numpy array encountered)')
                try:
                    os.remove(cache_filename)
                except Exception:
                    pass
                continue
        break
    return P_hat


def compute_normalized_inverse_propensity_score_weights(P, M,
                                                        scale_to_nnz=True):
    W = np.zeros(P.shape)
    M_one_mask = (M == 1)
    nnz = M_one_mask.sum()
    if nnz > 0:
        W[M_one_mask] = 1. / P[M_one_mask]
        if scale_to_nnz:
            W[M_one_mask] *= nnz / W[M_one_mask].sum()
        else:
            W[M_one_mask] /= W[M_one_mask].sum()
    return W


def compute_and_save_propensity_scores_rowcf(M, row_cf_k, row_cf_thres,
                                             approx, verbose=False,
                                             cache_prefix=''):
    if verbose:
        print('Estimating propensity scores via row cf knn...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        if approx:
            prefix = 'rowcf_approx'
        else:
            prefix = 'rowcf'
        cache_filename = os.path.join(cache_dir,
                                      '%s%s_k%d_t%f_'
                                      % (cache_prefix, prefix, row_cf_k,
                                         row_cf_thres)
                                      + memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            os.makedirs(cache_dir, exist_ok=True)
            if approx:
                tic = time.time()
                P_hat = row_cf_approx(M, row_cf_k, row_cf_thres)
                elapsed = time.time() - tic
            else:
                tic = time.time()
                P_hat = row_cf(M, row_cf_k, row_cf_thres)
                elapsed = time.time() - tic
            try:
                np.savetxt(cache_filename, P_hat)
                np.savetxt(cache_filename[:-4] + '_time.txt',
                           np.array(elapsed).reshape(1, -1))
            except Exception:
                pass
        else:
            try:
                P_hat = np.loadtxt(cache_filename)
                if P_hat.shape[0] != m or P_hat.shape[1] != n:
                    print('*** WARNING: Recomputing propensity scores '
                          + '(mismatched dimensions in cached file)')
                    try:
                        os.remove(cache_filename)
                    except Exception:
                        pass
                    continue
            except ValueError:
                print('*** WARNING: Recomputing propensity scores '
                      + '(malformed numpy array encountered)')
                try:
                    os.remove(cache_filename)
                except Exception:
                    pass
                continue
        break
    return P_hat


def compute_and_save_propensity_scores_naive_bayes(X, mcar_data,
                                                   min_prob, verbose=False,
                                                   cache_prefix=''):
    if verbose:
        print('Estimating propensity scores via logistic regression...')

    m, n = X.shape
    while True:
        memoize_hash = hashlib.sha256(X.data.tobytes())
        cache_filename = os.path.join(cache_dir,
                                      cache_prefix
                                      + 'nb_p%f_' % min_prob
                                      + memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            os.makedirs(cache_dir, exist_ok=True)
            tic = time.time()
            P_hat = estimate_propensities_naive_bayes(X, mcar_data,
                                                      min_prob)
            elapsed = time.time() - tic
            try:
                np.savetxt(cache_filename, P_hat)
                np.savetxt(cache_filename[:-4] + '_time.txt',
                           np.array(elapsed).reshape(1, -1))
            except Exception:
                pass
        else:
            try:
                P_hat = np.loadtxt(cache_filename)
                if P_hat.shape[0] != m or P_hat.shape[1] != n:
                    print('*** WARNING: Recomputing propensity scores '
                          + '(mismatched dimensions in cached file)')
                    try:
                        os.remove(cache_filename)
                    except Exception:
                        pass
                    continue
            except ValueError:
                print('*** WARNING: Recomputing propensity scores '
                      + '(malformed numpy array encountered)')
                try:
                    os.remove(cache_filename)
                except Exception:
                    pass
                continue
        break
    return P_hat


def compute_and_save_propensity_scores_log_reg(M, user_features, item_features,
                                               min_prob, verbose=False,
                                               cache_prefix=''):
    if verbose:
        print('Estimating propensity scores via logistic regression...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        cache_filename = os.path.join(cache_dir,
                                      cache_prefix
                                      + 'lr_p%f_' % min_prob
                                      + memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            os.makedirs(cache_dir, exist_ok=True)
            tic = time.time()
            P_hat = estimate_propensities_log_reg(M, user_features,
                                                  item_features,
                                                  min_prob)
            elapsed = time.time() - tic
            try:
                np.savetxt(cache_filename, P_hat)
                np.savetxt(cache_filename[:-4] + '_time.txt',
                           np.array(elapsed).reshape(1, -1))
            except Exception:
                pass
        else:
            try:
                P_hat = np.loadtxt(cache_filename)
                if P_hat.shape[0] != m or P_hat.shape[1] != n:
                    print('*** WARNING: Recomputing propensity scores '
                          + '(mismatched dimensions in cached file)')
                    try:
                        os.remove(cache_filename)
                    except Exception:
                        pass
                    continue
            except ValueError:
                print('*** WARNING: Recomputing propensity scores '
                      + '(malformed numpy array encountered)')
                try:
                    os.remove(cache_filename)
                except Exception:
                    pass
                continue
        break
    return P_hat


def estimate_propensities_naive_bayes(x_train, x_mar, min_prob=0.05):
    """
    The propensity score estimation strategy given by equation (18) of
    Schnabel et al 2016.

    Reference:

        Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and
        Thorsten Joachims.Recommendations as treatments: Debiasing learning and
        evaluation. In International Conferenceon Machine Learning, pages
        1670–1679, 2016

    WARNING: assumes that 0 means missing
    """
    if type(x_train) != csr_matrix:
        x_train = csr_matrix(x_train)
    if type(x_mar) != csr_matrix:
        x_mar = csr_matrix(x_mar)

    r_array = np.unique(x_train.data)
    num_train_data = len(x_train.data)
    num_mar_data = len(x_mar.data)

    p_o = num_train_data / np.float(x_train.shape[0] * x_train.shape[1])

    r_prob_dict = dict()
    for r in r_array:
        r_prob_dict[r] = ((x_train.data == r).sum() / num_train_data,
                          (x_mar.data == r).sum() / num_mar_data)

    P_hat = np.zeros(x_train.shape)
    for i in range(x_train.shape[0]):
        for idx, j in enumerate(x_train.indices[x_train.indptr[i]:
                                                x_train.indptr[i+1]]):
            pro, pr = r_prob_dict[x_train.data[x_train.indptr[i]:
                                               x_train.indptr[i+1]][idx]]
            if pr > 0:
                P_hat[i, j] = pro * p_o / pr

    # note: unobserved entries are not filled in since we are not going to be
    # weighting them anyways during inverse propensity score weighting

    P_hat = np.maximum(P_hat, min_prob)
    return P_hat


def estimate_propensities_log_reg(revealed_ratings,
                                  user_features,
                                  item_features,
                                  min_prob=0.05):
    num_users, num_items = revealed_ratings.shape
    feature_vectors = []
    labels = []
    user_one_hot = one_hot(np.arange(num_users))
    item_one_hot = one_hot(np.arange(num_items))
    for u in range(num_users):
        for i in range(num_items):
            feature_vector = np.concatenate((user_one_hot[u],
                                             item_one_hot[i],
                                             user_features[u],
                                             item_features[i]))
            feature_vectors.append(feature_vector)
            if revealed_ratings[u, i] > 0:
                labels.append(1)
            else:
                labels.append(0)
    feature_vectors = np.array(feature_vectors)
    labels = np.array(labels)

    # disable intercept since we explicitly add user-specific and, separately,
    # item-specific intercept terms to the feature vectors
    clf = LogisticRegressionCV(Cs=5, cv=5, scoring='neg_log_loss',
                               solver='lbfgs', fit_intercept=False,
                               random_state=0, n_jobs=1)

    clf.fit(feature_vectors, labels)
    predictions = clf.predict_proba(feature_vectors)[:, 1]
    P_hat = np.maximum(predictions.reshape((num_users, num_items)), min_prob)
    return P_hat


def compute_propensity_scores(M, propensity_scores, shape, one_bit_mc_tau,
                              one_bit_mc_gamma, one_bit_mc_max_rank,
                              one_bit_mc_approx_lr, row_cf_k, row_cf_thres,
                              approx, verbose, user_features=None,
                              item_features=None, min_prob=.05,
                              mcar_data=None, cache_prefix=''):
    if type(propensity_scores) == np.ndarray:
        return propensity_scores
    elif propensity_scores is None:
        return np.ones(shape)
    elif propensity_scores == 'naive bayes':
        return compute_and_save_propensity_scores_naive_bayes(
            M, mcar_data, min_prob, verbose=verbose, cache_prefix=cache_prefix)
    elif propensity_scores == 'logistic regression':
        return compute_and_save_propensity_scores_log_reg(
            M, user_features, item_features, min_prob, verbose=verbose,
            cache_prefix=cache_prefix)
    elif propensity_scores == '1bitmc':
        return compute_and_save_propensity_scores_1bitmc(
            M, one_bit_mc_tau, one_bit_mc_gamma, one_bit_mc_max_rank,
            one_bit_mc_approx_lr, approx, verbose=verbose,
            cache_prefix=cache_prefix)
    elif propensity_scores == '1bitmc_mod':
        if approx:
            raise NotImplementedError
        else:
            return compute_and_save_propensity_scores_1bitmc_mod(
                M, one_bit_mc_tau, one_bit_mc_gamma, one_bit_mc_max_rank,
                verbose=verbose, cache_prefix=cache_prefix)
    elif propensity_scores == 'rowcf':
        return compute_and_save_propensity_scores_rowcf(
            M, row_cf_k, row_cf_thres, approx, verbose=verbose,
            cache_prefix=cache_prefix)
    elif propensity_scores == '1bitmc-rowcf':
        P_hat_pre = compute_and_save_propensity_scores_1bitmc(
            M, one_bit_mc_tau, one_bit_mc_gamma, one_bit_mc_max_rank,
            one_bit_mc_approx_lr, approx, verbose=verbose,
            cache_prefix=cache_prefix)
        return compute_and_save_propensity_scores_rowcf(
            P_hat_pre, row_cf_k, row_cf_thres, approx, verbose=verbose,
            cache_prefix=cache_prefix)
    else:
        raise Exception('Unknown weight method: ' + propensity_scores)


# https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
def one_hot(data):
    encoding = np.zeros((data.size, data.max() + 1))
    encoding[np.arange(data.size), data] = 1
    return encoding


def linux_use_all_cores():
    # prevent numpy/scipy/etc from only using a single processor; see:
    # https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
    # (note that this is unix/linux only and should silently error on other
    # platforms)
    try:
        call(['taskset', '-p', '0x%s'
              % ('f' * int(np.ceil(os.cpu_count() / 4))),
              '%d' % os.getpid()], stdout=DEVNULL, stderr=DEVNULL)
    except Exception:
        pass
