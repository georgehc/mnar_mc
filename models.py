"""
Wrappers for some matrix completion algorithms to make them work with Surprise

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (weima@cmu.edu)
"""
import json
import hashlib
import numpy as np
import os
from scipy.sparse import csr_matrix
from soft_impute_ALS import WeightedSoftImpute_ALS
from surprise import AlgoBase, PredictionImpossible
from subprocess import DEVNULL, call
from mc_algorithms import one_bit_MC_fully_observed, std_logistic_function, \
        grad_std_logistic_function, weighted_softimpute, \
        one_bit_MC_mod_fully_observed, mod_logistic_function, \
        grad_mod_logistic_function, approx_doubly_weighted_trace_norm, \
        weighted_max_norm
from expomf import ExpoMF


cache_dir = 'cache_propensity_estimates'

# prevent numpy/scipy/etc from only using a single processor; see:
# https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
# (note that this is unix/linux only and should silently error on other
# platforms)
call(['taskset', '-p', '0x%s' % ('f' * int(np.ceil(os.cpu_count() / 4))),
      '%d' % os.getpid()], stdout=DEVNULL, stderr=DEVNULL)


class WeightedSoftImputeWrapper(AlgoBase):
    def __init__(self, max_rank, lmbda, min_value=None, max_value=None,
                 max_iter=200, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.min_value = min_value
        self.max_value = max_value
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.verbose = verbose

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores is None:
            P_hat = np.ones((trainset.n_users, trainset.n_items))
        elif self.propensity_scores == '1bitmc':
            P_hat = compute_and_save_propensity_scores_1bitmc(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        elif self.propensity_scores == '1bitmc_mod':
            P_hat = compute_and_save_propensity_scores_1bitmc_mod(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
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
                 one_bit_mc_gamma=1., verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.min_value = min_value
        self.max_value = max_value
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.verbose = verbose

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores is None:
            P_hat = np.ones((trainset.n_users, trainset.n_items))
        elif self.propensity_scores == '1bitmc':
            P_hat = compute_and_save_propensity_scores_1bitmc(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        elif self.propensity_scores == '1bitmc_mod':
            P_hat = compute_and_save_propensity_scores_1bitmc_mod(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
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
                 lmbda=10, max_iter=30, propensity_scores=None,
                 one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., random_state=None, verbose=False,
                 trace_norm_weighting=True):
        AlgoBase.__init__(self)
        self.min_value = min_value
        self.max_value = max_value
        self.n_components = n_components
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.random_state = random_state
        self.verbose = verbose
        self.trace_norm_weighting = trace_norm_weighting

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores is None:
            P_hat = np.ones((trainset.n_users, trainset.n_items))
        elif self.propensity_scores == '1bitmc':
            P_hat = compute_and_save_propensity_scores_1bitmc(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        elif self.propensity_scores == '1bitmc_mod':
            P_hat = compute_and_save_propensity_scores_1bitmc_mod(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        self.predictions = \
            approx_doubly_weighted_trace_norm(
                X, M, W, self.n_components,
                self.lmbda,
                min_value=self.min_value,
                max_value=self.max_value,
                opt_max_iter=self.max_iter,
                random_state=self.random_state,
                trace_norm_weighting=self.trace_norm_weighting)

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
    def __init__(self, min_value=None, max_value=None, n_components=20,
                 init_std=0.01, R=0.1, alpha=5, max_iter=100,
                 propensity_scores=None, one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1., verbose=False,
                 random_state=None):
        AlgoBase.__init__(self)
        self.min_value = min_value
        self.max_value = max_value
        self.n_components = n_components
        self.init_std = init_std
        self.R = R
        self.alpha = alpha
        self.max_iter = max_iter
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.verbose = verbose
        self.random_state = random_state

        if self.alpha is None:
            self.alpha = max(abs(min_value), abs(max_value))

        if min_value is not None and max_value is not None:
            # ignore user-supplied alpha
            self.alpha = max(abs(min_value), abs(max_value))
            if self.verbose:
                print('*** WARNING: Ignoring user-supplied alpha %f; '
                      % alpha
                      +
                      'using %f instead based on user-supplied min/max values'
                      % self.alpha)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores is None:
            P_hat = np.ones((trainset.n_users, trainset.n_items))
        elif self.propensity_scores == '1bitmc':
            P_hat = compute_and_save_propensity_scores_1bitmc(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        elif self.propensity_scores == '1bitmc_mod':
            P_hat = compute_and_save_propensity_scores_1bitmc_mod(
                        M, self.one_bit_mc_tau, self.one_bit_mc_gamma,
                        self.one_bit_mc_max_rank, verbose=self.verbose)
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        W = compute_normalized_inverse_propensity_score_weights(P_hat, M)
        X_hat = \
            weighted_max_norm(X, M, W,
                              opt_max_iter=self.max_iter,
                              n_components=self.n_components,
                              init_std=self.init_std,
                              R=self.R,
                              alpha=self.alpha,
                              random_state=self.random_state)

        # only clip afterward since otherwise for correctness we would have to
        # carefully change the optimization (note that the optimization is over
        # factorization terms U and V and not over X, and there is already
        # a constraint that the maximum absolute value of any entry in X be
        # bounded by alpha)
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


def compute_and_save_propensity_scores_1bitmc(M, one_bit_mc_tau,
                                              one_bit_mc_gamma,
                                              one_bit_mc_max_rank,
                                              verbose=False):
    if verbose:
        print('Estimating propensity scores via matrix completion...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        memoize_hash.update(
                json.dumps([one_bit_mc_tau,
                            one_bit_mc_gamma,
                            one_bit_mc_max_rank]).encode('utf-8'))
        cache_filename = os.path.join(cache_dir,
                                      memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
            os.makedirs(cache_dir, exist_ok=True)
            P_hat = one_bit_MC_fully_observed(M, std_logistic_function,
                                              grad_std_logistic_function,
                                              one_bit_mc_tau, one_bit_mc_gamma,
                                              max_rank=one_bit_mc_max_rank)
            try:
                np.savetxt(cache_filename, P_hat)
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
                                                  verbose=False):
    if verbose:
        print('Estimating propensity scores via matrix completion...')

    m, n = M.shape
    while True:
        memoize_hash = hashlib.sha256(M.data.tobytes())
        memoize_hash.update(
                json.dumps(['1bitmc-mod',
                            one_bit_mc_tau,
                            one_bit_mc_gamma,
                            one_bit_mc_max_rank]).encode('utf-8'))
        cache_filename = os.path.join(cache_dir,
                                      memoize_hash.hexdigest()
                                      + '.txt')
        if not os.path.isfile(cache_filename):
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
            try:
                np.savetxt(cache_filename, P_hat)
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


def compute_normalized_inverse_propensity_score_weights(P, M):
    W = np.zeros(P.shape)
    M_one_mask = (M == 1)
    num_observations = M_one_mask.sum()
    if num_observations > 0:
        W[M_one_mask] = 1. / P[M_one_mask]
        W[M_one_mask] *= num_observations / W[M_one_mask].sum()
    return W
