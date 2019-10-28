import json
import hashlib
import numpy as np
import os
cimport numpy as np
from scipy.sparse import csr_matrix
from surprise import AlgoBase, PredictionImpossible
from surprise.utils import get_rng
from mc_algorithms import one_bit_MC_fully_observed, std_logistic_function, \
        grad_std_logistic_function, one_bit_MC_mod_fully_observed, \
        mod_logistic_function, grad_mod_logistic_function
from models import cache_dir


class WeightedSVD(AlgoBase):
    """
    This is a slightly modified version of the Surprise SVD class to enable
    weighting of the individual matrix entries.

    The parameters for initialization are the same as SVD except now there
    are a few extras:
    - propensity_scores: '1bitmc' or a user-supplied propensity score matrix
    - one_bit_mc_max_rank: for running 1bitmc, what to use as the max rank
        during optimization (choose None to use full SVD or specify an integer
        to use randomized SVD)
    - one_bit_mc_tau: ask for nuclear norm of the 1bitMC parameter matrix to be
        at most tau*sqrt( number of rows * number of columns )
    - one_bit_mc_gamma: ask for the entry-wise max norm of the 1bitMC parameter
        matrix to be at most gamma
    """

    def __init__(self, n_factors=100, n_epochs=20, biased=True, init_mean=0,
                 init_std_dev=.1, lr_all=.005,
                 reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 random_state=None, verbose=False,
                 propensity_scores='1bitmc', one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1.):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.biased = biased
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, _ in trainset.all_ratings():
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

        weights = 1 / P_hat
        sparse_row_indices, sparse_col_indices, sparse_data = \
            zip(*[(u, i, weights[u, i]) for u, i, r in trainset.all_ratings()])
        self.weights_sparse = \
            csr_matrix((sparse_data, (sparse_row_indices, sparse_col_indices)),
                       shape=(trainset.n_users, trainset.n_items)).tocsr()

        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # OK, let's breath. I've seen so many different implementation of this
        # algorithm that I just not sure anymore of what it should do. I've
        # implemented the version as described in the BellKor papers (RS
        # Handbook, etc.). Mymedialite also does it this way. In his post
        # however, Funk seems to implicitly say that the algo looks like this
        # (see reg below):
        # for f in range(n_factors):
        #       for _ in range(n_iter):
        #           for u, i, r in all_ratings:
        #               err = r_ui - <p[u, :f+1], q[i, :f+1]>
        #               update p[u, f]
        #               update q[i, f]
        # which is also the way https://github.com/aaw/IncrementalSVD.jl
        # implemented it.
        #
        # Funk: "Anyway, this will train one feature (aspect), and in
        # particular will find the most prominent feature remaining (the one
        # that will most reduce the error that's left over after previously
        # trained features have done their best). When it's as good as it's
        # going to get, shift it onto the pile of done features, and start a
        # new one. For efficiency's sake, cache the residuals (all 100 million
        # of them) so when you're training feature 72 you don't have to wait
        # for predictRating() to re-compute the contributions of the previous
        # 71 features. You will need 2 Gig of ram, a C compiler, and good
        # programming habits to do this."

        # A note on cythonization: I haven't dived into the details, but
        # accessing 2D arrays like pu using just one of the indices like pu[u]
        # is not efficient. That's why the old (cleaner) version can't be used
        # anymore, we need to compute the dot products by hand, and update
        # user and items factors by iterating over all factors...

        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, f
        cdef double r, err, dot, puf, qif
        cdef double global_mean = self.trainset.global_mean

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        rng = get_rng(self.random_state)

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))

        if not self.biased:
            global_mean = 0

        weights = self.weights_sparse

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = (r - (global_mean + bu[u] + bi[i] + dot)) * weights[u, i]

                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if self.biased:
            est = self.trainset.global_mean

            if known_user:
                est += self.bu[u]

            if known_item:
                est += self.bi[i]

            if known_user and known_item:
                est += np.dot(self.qi[i], self.pu[u])

        else:
            if known_user and known_item:
                est = np.dot(self.qi[i], self.pu[u])
            else:
                raise PredictionImpossible('User and item are unkown.')

        return est


class WeightedSVDpp(AlgoBase):
    """
    This is a slightly modified version of the Surprise SVDpp class to enable
    weighting of the individual matrix entries.

    The parameters for initialization are the same as SVDpp except now there
    are a few extras:
    - propensity_scores: '1bitmc' or a user-supplied propensity score matrix
    - one_bit_mc_max_rank: for running 1bitmc, what to use as the max rank
        during optimization (choose None to use full SVD or specify an integer
        to use randomized SVD)
    - one_bit_mc_tau: ask for nuclear norm of the 1bitMC parameter matrix to be
        at most tau*sqrt( number of rows * number of columns )
    - one_bit_mc_gamma: ask for the entry-wise max norm of the 1bitMC parameter
        matrix to be at most gamma
    """

    def __init__(self, n_factors=20, n_epochs=20, init_mean=0, init_std_dev=.1,
                 lr_all=.007, reg_all=.02, lr_bu=None, lr_bi=None, lr_pu=None,
                 lr_qi=None, lr_yj=None, reg_bu=None, reg_bi=None, reg_pu=None,
                 reg_qi=None, reg_yj=None, random_state=None, verbose=False,
                 propensity_scores='1bitmc', one_bit_mc_max_rank=None,
                 one_bit_mc_tau=1., one_bit_mc_gamma=1.):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all
        self.random_state = random_state
        self.verbose = verbose
        self.propensity_scores = propensity_scores
        self.one_bit_mc_max_rank = one_bit_mc_max_rank
        self.one_bit_mc_tau = one_bit_mc_tau
        self.one_bit_mc_gamma = one_bit_mc_gamma

        AlgoBase.__init__(self)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        M = np.zeros((trainset.n_users, trainset.n_items))
        for u, i, _ in trainset.all_ratings():
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
                            os.remove(cache_filename)
                            continue
                    except ValueError:
                        print('*** WARNING: Recomputing propensity scores '
                              + '(malformed numpy array encountered)')
                        os.remove(cache_filename)
                        continue
                break
        else:
            raise Exception('Unknown weight method: ' + self.propensity_scores)
        self.propensity_estimates = P_hat

        if self.verbose:
            print('Running debiased matrix completion...')

        weights = 1 / P_hat
        sparse_row_indices, sparse_col_indices, sparse_data = \
            zip(*[(u, i, weights[u, i]) for u, i, r in trainset.all_ratings()])
        self.weights_sparse = \
            csr_matrix((sparse_data, (sparse_row_indices, sparse_col_indices)),
                       shape=(trainset.n_users, trainset.n_items)).tocsr()

        self.sgd(trainset)

        return self

    def sgd(self, trainset):

        # user biases
        cdef np.ndarray[np.double_t] bu
        # item biases
        cdef np.ndarray[np.double_t] bi
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi
        # item implicit factors
        cdef np.ndarray[np.double_t, ndim=2] yj

        cdef int u, i, j, f
        cdef double r, err, dot, puf, qif, sqrt_Iu, _
        cdef double global_mean = self.trainset.global_mean
        cdef np.ndarray[np.double_t] u_impl_fdb

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj

        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)

        rng = get_rng(self.random_state)

        pu = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_users, self.n_factors))
        qi = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        yj = rng.normal(self.init_mean, self.init_std_dev,
                        (trainset.n_items, self.n_factors))
        u_impl_fdb = np.zeros(self.n_factors, np.double)

        weights = self.weights_sparse

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print(" processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():

                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicit feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu

                # compute current error
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                err = (r - (global_mean + bu[u] + bi[i] + dot)) * weights[u, i]

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                         reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu -
                                             reg_yj * yj[j, f])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):

        est = self.trainset.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _)
                               in self.trainset.ur[u]) / np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est
