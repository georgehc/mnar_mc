"""
This file contains the SoftImpute-ALS library by github user `andrewdalex`
(https://github.com/andrewdalex/SoftImpute-ALS) along with a modified variant
that weights the entries of the rating reconstruction sum-of-squared-errors
term; this modification allows for inverse probability weighting. The modified
variant is by George H. Chen (georgechen@cmu.edu).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import numpy.linalg as npla
import matplotlib.pyplot as plt
import time


class SoftImpute_ALS:

    def __init__(self, k, R, verbose):
        self._R = R
        self._k = k
        self._m, self._n = self._R.shape
        self._bootstrap(k)
        self._Rt = self._R.T
        self._X_star = np.zeros((self._m, self._n))
        self._Xt_star = np.zeros((self._n, self._m))
        self._verbose = verbose

    def _bootstrap(self, k):
        self._U = np.zeros((self._m, k))
        self._V = np.zeros((self._n, k))
        self._Dsq = np.eye(k)
        self._U[:, :] = np.random.randn(self._m, k)
        self._U, _, _ = np.linalg.svd(self._U, False)
        self._U_old = np.zeros((self._m, k))
        self._V_old = np.zeros((self._n, k))
        self._Dsq_old = np.eye(k)

    def _frob(self):
        denom = np.trace((self._Dsq_old**4))
        utu = np.dot(self._Dsq_old**2, (self._U_old.T.dot(self._U)))
        vtv = np.dot(self._Dsq**2, (self._V.T.dot(self._V_old)))
        uvprod = np.trace(utu.dot(vtv))
        num = denom + np.trace((self._Dsq ** 4)) - 2*uvprod
        return num / denom

    def _compute_cost(self):
        pattern = self._R != 0
        proj_ABt = pattern.multiply((self._A).dot(self._B.T))
        cost = splinalg.norm(proj_ABt - self._R, 'fro') ** 2
        norm_A = self._Lambda * (np.linalg.norm(self._A, 'fro')) ** 2
        norm_B = self._Lambda * (np.linalg.norm(self._B, 'fro')) ** 2
        return cost + norm_A + norm_B

    def fit(self, k=40, thresh=1e-05, Lambda=20, maxit=50, plot_conv=None,
            plot_time=None):
        verbose = self._verbose

        if (k != self._k):
            self._bootstrap(k)
            self._k = k

        self._A = np.dot(self._U, self._Dsq)
        self._B = np.dot(self._V, self._Dsq)

        ratio = 1
        itr = 0
        self._Lambda = Lambda

        if plot_conv is not None:
            x_plot = []
            y_plot = []

        if plot_time is not None:
            t = []

        while((thresh < ratio) and (itr < maxit)):
            if verbose:
                print("=== Starting Iteration " + str(itr) + " ===")
            t0 = time.clock()
            itr = itr + 1
            self._U_old[:, :] = self._U
            self._V_old[:, :] = self._V
            self._Dsq_old[:, :] = self._Dsq

            # B step
            pattern = (self._R != 0)
            ABt = self._A.dot(self._B.T)
            proj_ABt = pattern.multiply(ABt)
            self._X_star[:, :] = self._R.multiply(pattern) - proj_ABt + ABt
            left_side = self._Dsq**2 + (self._Lambda * np.eye(k))
            right_side = np.dot(self._Dsq, np.dot(self._U.T, self._X_star))
            B_tilde = npla.solve(left_side, right_side)
            self._V[:, :], D, _ = npla.svd(np.dot(B_tilde.T, self._Dsq), False)
            self._Dsq[:, :] = np.diag(np.sqrt(D))
            self._B[:, :] = np.dot(self._V, self._Dsq)

            # A step
            pattern = (self._R.T != 0)
            BAt = self._B.dot(self._A.T)
            proj_BAt = pattern.multiply(BAt)
            self._Xt_star[:, :] = self._Rt.multiply(pattern) - proj_BAt + BAt
            left_side = self._Dsq**2 + (self._Lambda * np.eye(k))
            right_side = np.dot(self._Dsq, np.dot(self._V.T, self._Xt_star))
            A_tilde = npla.solve(left_side, right_side)

            self._U[:, :], d, _ = npla.svd(np.dot(A_tilde.T, self._Dsq), False)
            self._Dsq[:, :] = np.diag(np.sqrt(d))
            self._A[:, :] = np.dot(self._U, self._Dsq)

            t1 = time.clock()
            ratio = self._frob()
            cost = self._compute_cost()

            if verbose:
                print("Cost => " + str(cost))
                print("Ratio => " + str(ratio))

            if plot_conv is not None:
                y_plot.append(cost)
                x_plot.append(itr)

            if plot_time is not None:
                t.append(t1 - t0)

        # Final Step: Output Solution
        M = self._X_star.dot(self._V)
        self._U[:, :], d, v = npla.svd(M, False)
        self._V[:, :] = np.dot(self._V, v)
        d = np.fmax(d - Lambda, 0)
        self._Dsq = np.diag(d[d > 0])
        k, _ = self._Dsq.shape
        self._U = self._U[:, :k]
        self._Dsq = self._Dsq[:k]
        self._V = self._V[:, :k]

        if plot_conv is not None:
            plt.figure()
            plt.plot(x_plot, y_plot, 'sg')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Computed Cost')
            plt.title('Lambda = 5')
            plt.savefig(plot_conv)

        if plot_time is not None:
            plt.figure()
            plt.plot(t, 'sg')
            plt.ylabel('Single Iteration runtime')
            plt.xlabel('Iteration number')
        return

    def compute_rmse(self, R_test):
        prediction = self._U.dot(self._Dsq.dot(self._V.T))
        pattern = (R_test != 0)
        s = splinalg.norm(pattern.multiply(prediction) - R_test, 'fro') ** 2
        return np.sqrt(s / R_test.nnz)

    def get_UVD(self):
        # return copies so as to not corrupt internal structures
        return (self._U.copy(), self._V.copy(), self._Dsq.copy())


class WeightedSoftImpute_ALS:

    def __init__(self, k, R, W_dense, verbose):
        self._R = R
        self._k = k
        self._m, self._n = self._R.shape
        self._bootstrap(k)
        self._Rt = self._R.T
        self._X_star = np.zeros((self._m, self._n))
        self._Xt_star = np.zeros((self._n, self._m))
        self._W_dense = W_dense
        self._ptwise_sqrt_W_sparse = (R != 0).multiply(W_dense**0.5)
        self._verbose = verbose

    def _bootstrap(self, k):
        self._U = np.zeros((self._m, k))
        self._V = np.zeros((self._n, k))
        self._Dsq = np.eye(k)
        self._U[:, :] = np.random.randn(self._m, k)
        self._U, _, _ = np.linalg.svd(self._U, False)
        self._U_old = np.zeros((self._m, k))
        self._V_old = np.zeros((self._n, k))
        self._Dsq_old = np.eye(k)

    def _frob(self):
        denom = np.trace((self._Dsq_old**4))
        utu = np.dot(self._Dsq_old**2, (self._U_old.T.dot(self._U)))
        vtv = np.dot(self._Dsq**2, (self._V.T.dot(self._V_old)))
        uvprod = np.trace(utu.dot(vtv))
        num = denom + np.trace((self._Dsq ** 4)) - 2*uvprod
        return num / denom

    def _compute_cost(self):
        pattern = self._R != 0
        proj_ABt = pattern.multiply((self._A).dot(self._B.T))
        cost = splinalg.norm((proj_ABt - self._R).multiply(
                self._ptwise_sqrt_W_sparse), 'fro') ** 2
        norm_A = self._Lambda * (np.linalg.norm(self._A, 'fro')) ** 2
        norm_B = self._Lambda * (np.linalg.norm(self._B, 'fro')) ** 2
        return cost + norm_A + norm_B

    def fit(self, k=40, thresh=1e-05, Lambda=20, maxit=50, plot_conv=None,
            plot_time=None):
        verbose = self._verbose

        if (k != self._k):
            self._bootstrap(k)
            self._k = k

        self._A = np.dot(self._U, self._Dsq)
        self._B = np.dot(self._V, self._Dsq)

        ratio = 1
        itr = 0
        self._Lambda = Lambda

        if plot_conv is not None:
            x_plot = []
            y_plot = []

        if plot_time is not None:
            t = []

        while((thresh < ratio) and (itr < maxit)):
            if verbose:
                print("=== Starting Iteration " + str(itr) + " ===")
            t0 = time.clock()
            itr = itr + 1
            self._U_old[:, :] = self._U
            self._V_old[:, :] = self._V
            self._Dsq_old[:, :] = self._Dsq

            # B step
            pattern = (self._R != 0)
            ABt = self._A.dot(self._B.T)
            proj_ABt = pattern.multiply(ABt)

            self._X_star[:, :] = self._R.multiply(pattern) - proj_ABt + ABt
            B_tilde = np.zeros((k, self._n))
            diag_vec = np.diag(self._Dsq)
            UD = np.multiply(self._U, diag_vec)
            for j in range(self._n):
                shared = np.multiply(UD.T, self._W_dense[:, j])
                left_side = np.dot(shared, UD) + (self._Lambda * np.eye(k))
                right_side = np.dot(shared, self._X_star[:, j])
                B_tilde[:, j] = npla.solve(left_side, right_side)

            self._V[:, :], D, _ = npla.svd(np.multiply(B_tilde.T, diag_vec),
                                           False)
            self._Dsq[:, :] = np.diag(np.sqrt(D))
            self._B[:, :] = np.dot(self._V, self._Dsq)

            # A step
            pattern = (self._R.T != 0)
            BAt = self._B.dot(self._A.T)
            proj_BAt = pattern.multiply(BAt)
            self._Xt_star[:, :] = self._Rt.multiply(pattern) - proj_BAt + BAt
            A_tilde = np.zeros((k, self._m))
            diag_vec = np.diag(self._Dsq)
            VD = np.multiply(self._V, diag_vec)
            for j in range(self._m):
                shared = np.multiply(VD.T, self._W_dense[j])
                left_side = np.dot(shared, VD) + (self._Lambda * np.eye(k))
                right_side = np.dot(shared, self._Xt_star[:, j])
                A_tilde[:, j] = npla.solve(left_side, right_side)

            self._U[:, :], d, _ = npla.svd(np.multiply(A_tilde.T, diag_vec),
                                           False)
            self._Dsq[:, :] = np.diag(np.sqrt(d))
            self._A[:, :] = np.dot(self._U, self._Dsq)

            t1 = time.clock()
            ratio = self._frob()
            cost = self._compute_cost()
            if verbose:
                print("Cost => " + str(cost))
                print("Ratio => " + str(ratio))

            if plot_conv is not None:
                y_plot.append(cost)
                x_plot.append(itr)

            if plot_time is not None:
                t.append(t1 - t0)

        # Final Step: Output Solution
        M = self._X_star.dot(self._V)
        self._U[:, :], d, v = npla.svd(M, False)
        self._V[:, :] = np.dot(self._V, v)
        d = np.fmax(d - Lambda, 0)
        self._Dsq = np.diag(d[d > 0])
        k, _ = self._Dsq.shape
        self._U = self._U[:, :k]
        self._Dsq = self._Dsq[:k]
        self._V = self._V[:, :k]

        if plot_conv is not None:
            plt.figure()
            plt.plot(x_plot, y_plot, 'sg')
            plt.xlabel('Number of Iterations')
            plt.ylabel('Computed Cost')
            plt.title('Lambda = 5')
            plt.savefig(plot_conv)

        if plot_time is not None:
            plt.figure()
            plt.plot(t, 'sg')
            plt.ylabel('Single Iteration runtime')
            plt.xlabel('Iteration number')
        return

    def compute_rmse(self, R_test):
        prediction = self._U.dot(self._Dsq.dot(self._V.T))
        pattern = (R_test != 0)
        s = splinalg.norm(pattern.multiply(prediction) - R_test, 'fro') ** 2
        return np.sqrt(s / R_test.nnz)

    def get_UVD(self):
        # return copies so as to not corrupt internal structures
        return (self._U.copy(), self._V.copy(), self._Dsq.copy())


def rmse_rank_lambda_plot(R_train, R_test, ranks_to_try, lambdas_to_try):
    for l in lambdas_to_try:
        test_rmse = []
        train_rmse = []
        plot_name = "plots/rmse_Lambda_{}.jpg".format(l)
        for k in ranks_to_try:
            sals = SoftImpute_ALS(k, R_train)
            sals.fit(Lambda=l, maxit=100)
            train_rmse.append(sals.compute_rmse(R_train))
            test_rmse.append(sals.compute_rmse(R_test))
        plt.figure()
        plt.plot(ranks_to_try, test_rmse, 'sb', ranks_to_try, train_rmse, 'sg')
        plt.legend()
        plt.xlabel('Ranks')
        plt.ylabel('RMSE values with Lambda {}'.format(l))
        plt.savefig(plot_name)


def text_to_CSR(filename, m, n):
    raw_data = np.genfromtxt(filename, dtype=np.int32)
    users = raw_data[:, 0] - 1
    items = raw_data[:, 1] - 1
    ratings = raw_data[:, 2].astype(np.float64)
    R = sp.coo_matrix((ratings, (users, items)), shape=(m, n))
    return R.tocsr()


def main():
    # to run this, the movielens-100k dataset should be downloaded and placed
    # in the directory 'ml-100k/'
    num_users = 943
    num_items = 1682
    R_train = text_to_CSR('ml-100k/ub.base', num_users, num_items)
    R_test = text_to_CSR('ml-100k/ub.test', num_users, num_items)
    sals = SoftImpute_ALS(40, R_train)
    sals.fit(plot_conv="plots/conv_test.jpg", plot_time="plots/time_test.jpg")
    print(sals.compute_rmse(R_test))

    lambdas_to_try = \
        [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    ranks_to_try = [3, 5, 10, 15, 20, 25, 30, 35, 40, 50]
    rmse_rank_lambda_plot(R_train, R_test, ranks_to_try, lambdas_to_try)


if __name__ == "__main__":
    main()
