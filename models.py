"""
Wrappers for some matrix completion algorithms to make them work with Surprise

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (weima@cmu.edu)
"""
import json
import hashlib
import numpy as np
import os
from contextlib import redirect_stderr
from scipy.sparse import csr_matrix
from surprise import AlgoBase, PredictionImpossible
from soft_impute_ALS import SoftImpute_ALS, WeightedSoftImpute_ALS
from mc_algorithms import one_bit_MC_fully_observed, std_logistic_function, \
        grad_std_logistic_function, weighted_softimpute, \
        one_bit_MC_mod_fully_observed, mod_logistic_function, \
        grad_mod_logistic_function, DoublyWeightedTraceNorm, MaxNorm, \
        WeightedMaxNorm
with redirect_stderr(open(os.devnull, "w")):
    from fancyimpute import SoftImpute
from expomf import ExpoMF


cache_dir = 'cache_propensity_estimates'


class SoftImputeWrapper(AlgoBase):

    def __init__(self, max_rank, lmbda, max_iter=200, min_value=None,
                 max_value=None, verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.verbose = verbose
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        X_incomplete = np.nan*np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            X_incomplete[u, i] = r

        soft_impute = SoftImpute(shrinkage_value=self.lmbda,
                                 max_iters=self.max_iter,
                                 max_rank=self.max_rank,
                                 min_value=self.min_value,
                                 max_value=self.max_value,
                                 verbose=self.verbose)
        X_filled_normalized \
            = soft_impute.fit_transform(X_incomplete)
        self.predictions = X_filled_normalized
        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if known_user and known_item:
            return self.predictions[u, i]
        else:
            raise PredictionImpossible('User and item are unkown.')


class WeightedSoftImputeWrapper(AlgoBase):

    def __init__(self, max_rank, lmbda,
                 propensity_scores='1bitmc', max_iter=200,
                 min_value=None, max_value=None, one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1., verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.verbose = verbose
        self.min_value = min_value
        self.max_value = max_value
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.propensity_scores = propensity_scores

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        X = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1
            X[u, i] = r

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores == '1bitmc':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps([self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_fully_observed(
                            M, std_logistic_function,
                            grad_std_logistic_function,
                            self.one_bit_mc_tau,
                            self.one_bit_mc_gamma,
                            max_rank=self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        elif self.propensity_scores == '1bitmc_mod':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps(['1bitmc-mod',
                                    self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    one_minus_logistic_gamma \
                        = 1 - std_logistic_function(self.one_bit_mc_gamma)
                    link = lambda x: \
                        mod_logistic_function(x, self.one_bit_mc_gamma,
                                              one_minus_logistic_gamma)
                    grad_link = lambda x: \
                        grad_mod_logistic_function(x, self.one_bit_mc_gamma,
                                                   one_minus_logistic_gamma)
                                                           
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_mod_fully_observed(M, link, grad_link,
                                                      self.one_bit_mc_tau,
                                                      self.one_bit_mc_gamma,
                                                      max_rank=
                                                      self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        self.predictions = \
            weighted_softimpute(X, M, 1 / P_hat, self.lmbda, self.max_rank,
                                self.min_value, self.max_value)

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


class SoftImputeALSWrapper(AlgoBase):

    def __init__(self, max_rank, lmbda, max_iter=200, verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        sparse_row_indices, sparse_col_indices, sparse_data = \
            zip(*[(u, i, r) for u, i, r in trainset.all_ratings()])
        X_incomplete_csr = \
            csr_matrix((sparse_data, (sparse_row_indices, sparse_col_indices)),
                       shape=(trainset.n_users, trainset.n_items)).tocsr()

        sals = SoftImpute_ALS(self.max_rank, X_incomplete_csr,
                              verbose=self.verbose)
        sals.fit(Lambda=self.lmbda, maxit=self.max_iter)
        self.predictions = sals._U.dot(sals._Dsq.dot(sals._V.T))

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

    def __init__(self, max_rank, lmbda, propensity_scores='1bitmc',
                 max_iter=200, one_bit_mc_max_rank=None, one_bit_mc_tau=1.,
                 one_bit_mc_gamma=1., verbose=False):
        AlgoBase.__init__(self)

        self.max_rank = max_rank
        self.lmbda = lmbda
        self.max_iter = max_iter
        self.propensity_scores  = propensity_scores
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
        elif self.propensity_scores == '1bitmc':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps([self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_fully_observed(
                            M, std_logistic_function,
                            grad_std_logistic_function,
                            self.one_bit_mc_tau,
                            self.one_bit_mc_gamma,
                            max_rank=self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        elif self.propensity_scores == '1bitmc_mod':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps(['1bitmc-mod',
                                    self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    one_minus_logistic_gamma \
                        = 1 - std_logistic_function(self.one_bit_mc_gamma)
                    link = lambda x: \
                        mod_logistic_function(x, self.one_bit_mc_gamma,
                                              one_minus_logistic_gamma)
                    grad_link = lambda x: \
                        grad_mod_logistic_function(x, self.one_bit_mc_gamma,
                                                   one_minus_logistic_gamma)
                                                           
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_mod_fully_observed(M, link, grad_link,
                                                      self.one_bit_mc_tau,
                                                      self.one_bit_mc_gamma,
                                                      max_rank=
                                                      self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
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

        sals = WeightedSoftImpute_ALS(self.max_rank, X_incomplete_csr,
                                      1 / P_hat, self.verbose)
        sals.fit(Lambda=self.lmbda, maxit=self.max_iter)
        self.predictions = sals._U.dot(sals._Dsq.dot(sals._V.T))

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
    def __init__(self, n_components=20, max_iter=100, batch_size=1024,
                 init_std=0.01, n_jobs=8, random_state=None,
                 early_stopping=False, verbose=False):
        AlgoBase.__init__(self)
        self.clf = ExpoMF(n_components=n_components, max_iter=max_iter,
                          init_std=init_std, random_state=random_state)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r
        Y = csr_matrix(Y)
        self.clf.fit(Y)
        U, V = self.clf.theta, self.clf.beta
        X_hat = U.dot(V.T)
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


class WeightedTraceNormWrapper(AlgoBase):
    def __init__(self, k=3, alpha=2, lambda1=10, step_size=0.0001, n_epoch=30,
                 random_state=None):
        AlgoBase.__init__(self)
        self.k = k
        self.alpha = alpha
        self.lambda1 = lambda1
        self.step_size = step_size
        self.n_epoch = n_epoch
        self.random_state = random_state

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r

        self.clf = DoublyWeightedTraceNorm(Y, self.k, self.alpha, self.lambda1,
                                           max_iter=self.n_epoch,
                                           random_state=self.random_state)
        self.clf.fit()
        X_hat = self.clf.X_hat
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
    def __init__(self, k=3, alpha=2, lambda1=10, step_size=0.0001, n_epoch=30,
                 propensity_scores='1bitmc', one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1., verbose=False,
                 random_state=None):
        AlgoBase.__init__(self)
        self.k = k
        self.alpha = alpha
        self.lambda1 = lambda1
        self.step_size = step_size
        self.n_epoch = n_epoch
        self.propensity_scores  = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores == '1bitmc':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps([self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_fully_observed(
                            M, std_logistic_function,
                            grad_std_logistic_function,
                            self.one_bit_mc_tau,
                            self.one_bit_mc_gamma,
                            max_rank=self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        elif self.propensity_scores == '1bitmc_mod':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps(['1bitmc-mod',
                                    self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    one_minus_logistic_gamma \
                        = 1 - std_logistic_function(self.one_bit_mc_gamma)
                    link = lambda x: \
                        mod_logistic_function(x, self.one_bit_mc_gamma,
                                              one_minus_logistic_gamma)
                    grad_link = lambda x: \
                        grad_mod_logistic_function(x, self.one_bit_mc_gamma,
                                                   one_minus_logistic_gamma)
                                                           
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_mod_fully_observed(M, link, grad_link,
                                                      self.one_bit_mc_tau,
                                                      self.one_bit_mc_gamma,
                                                      max_rank=
                                                      self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r

        self.clf = DoublyWeightedTraceNorm(Y, self.k, self.alpha, self.lambda1,
                                           max_iter=self.n_epoch,
                                           random_state=self.random_state)
        self.clf.fit(P_hat)
        X_hat = self.clf.X_hat
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


class MaxNormWrapper(AlgoBase):
    def __init__(self, n_components=20, max_iter=100, 
                 init_std=0.01, R=0.1, alpha=5, random_state=None):
        AlgoBase.__init__(self)
        self.clf = MaxNorm(n_components=n_components, max_iter=max_iter,
                           init_std=init_std, R=R, alpha=alpha,
                           random_state=random_state)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r

        self.clf.fit(Y)
        self.predictions = self.clf.X_hat

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
    def __init__(self, n_components=20, max_iter=100,
                 propensity_scores='1bitmc', one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1.,
                 init_std=0.01, R=0.1, alpha=5, verbose=False,
                 random_state=None):
        AlgoBase.__init__(self)
        self.propensity_scores  = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma
        self.verbose = verbose
        self.clf = WeightedMaxNorm(n_components=n_components,
                                   max_iter=max_iter, init_std=init_std, R=R,
                                   alpha=alpha, random_state=random_state)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            M[u, i] = 1

        if type(self.propensity_scores) == np.ndarray:
            P_hat = self.propensity_scores
        elif self.propensity_scores == '1bitmc':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps([self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_fully_observed(
                            M, std_logistic_function,
                            grad_std_logistic_function,
                            self.one_bit_mc_tau,
                            self.one_bit_mc_gamma,
                            max_rank=self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        elif self.propensity_scores == '1bitmc_mod':
            if self.verbose:
                print('Estimating propensity scores via matrix completion...')

            while True:
                memoize_hash = hashlib.sha256(M.data.tobytes())
                memoize_hash.update(
                        json.dumps(['1bitmc-mod',
                                    self.one_bit_mc_tau,
                                    self.one_bit_mc_gamma,
                                    self.one_bit_mc_max_rank]).encode('utf-8'))
                cache_filename = os.path.join(cache_dir,
                                              memoize_hash.hexdigest()
                                              + '.txt')
                if not os.path.isfile(cache_filename):
                    one_minus_logistic_gamma \
                        = 1 - std_logistic_function(self.one_bit_mc_gamma)
                    link = lambda x: \
                        mod_logistic_function(x, self.one_bit_mc_gamma,
                                              one_minus_logistic_gamma)
                    grad_link = lambda x: \
                        grad_mod_logistic_function(x, self.one_bit_mc_gamma,
                                                   one_minus_logistic_gamma)
                                                           
                    os.makedirs(cache_dir, exist_ok=True)
                    P_hat = \
                        one_bit_MC_mod_fully_observed(M, link, grad_link,
                                                      self.one_bit_mc_tau,
                                                      self.one_bit_mc_gamma,
                                                      max_rank=
                                                      self.one_bit_mc_max_rank)
                    try:
                        np.savetxt(cache_filename, P_hat)
                    except:
                        pass
                else:
                    try:
                        P_hat = np.loadtxt(cache_filename)
                        if P_hat.shape[0] != trainset.n_users or \
                                P_hat.shape[1] != trainset.n_items:
                            print('*** WARNING: Recomputing propensity scores '
                                  + '(mismatched dimensions in cached file)')
                            try:
                                os.remove(cache_filename)
                            except:
                                pass
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        try:
                            os.remove(cache_filename)
                        except:
                            pass
                        continue
                break
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        Y = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, r in trainset.all_ratings():
            Y[u, i] = r

        self.clf.fit(Y, P_hat)
        self.predictions = self.clf.X_hat

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
