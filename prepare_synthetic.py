"""
Code for generating synthetic datasets

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
"""
import os
import numpy as np
from scipy.sparse import csr_matrix
from mc_algorithms import std_logistic_function


num_components = 5
num_users = 512
num_items = 512

useritemfeature_output_dir = 'useritemfeature'
cluster_output_dir = 'rowcluster'
steck_output_dir = 'steck'


def dense_to_sparse(X):
    m, n = X.shape
    return [(u, i, X[u, i]) for u in range(m)
            for i in range(n)
            if X[u, i] != 0]


def user_item_features_propensities_ratings(num_rows, num_cols, num_components,
                                            nu, P_bias, S_threshold):
    U = np.random.randn(num_rows, num_components)
    U = np.array([row / np.linalg.norm(row) for row in U])

    V = np.random.randn(num_cols, num_components)
    V = np.array([row / np.linalg.norm(row) for row in V])

    param_U = np.random.randn(num_components) / nu
    param_V = np.random.randn(num_components) / nu

    P = np.zeros((num_rows, num_cols))
    S = np.zeros((num_rows, num_cols))
    for u in range(num_rows):
        for i in range(num_cols):
            inner_prod = np.inner(U[u], param_U) \
                + np.inner(V[i], param_V)
            P[u, i] = \
                std_logistic_function(inner_prod + P_bias)
            if inner_prod <= S_threshold:
                S[u, i] = 1
            else:
                S[u, i] = 5

    return P, S, U, V


def cluster_true_propensities_true_ratings(num_rows, num_cols, num_components,
                                           eps=0.25):
    P = np.ones((num_rows, num_cols)) * eps
    col_seq = np.array_split(np.arange(num_cols), num_components)
    pi = np.ones(num_components) / num_components
    membership = np.random.choice(num_components, size=num_rows, replace=True,
                                  p=pi)
    for i in range(num_rows):
        P[i, col_seq[membership[i]]] = 1 - eps

    cluster_ratings = 1 + np.random.rand(num_components, num_cols) * 4
    S = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        S[i] = cluster_ratings[membership[i]]

    return P, S


def steck_propensities_ratings(num_rows, num_cols, p=0.5):
    u_seq = np.array_split(range(num_rows), 2)
    i_seq = np.array_split(range(num_cols), 3)

    P = np.ones((num_rows, num_cols)) * p
    P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] *= 1
    P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] *= 0.1
    P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] *= 0.5
    P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] *= 0.1
    P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] *= 1
    P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] *= 0.5

    S = np.zeros((num_rows, num_cols))
    S[u_seq[0][0]:u_seq[0][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] = 5
    S[u_seq[0][0]:u_seq[0][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] = 1
    S[u_seq[0][0]:u_seq[0][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] = 3
    S[u_seq[1][0]:u_seq[1][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] = 1
    S[u_seq[1][0]:u_seq[1][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] = 5
    S[u_seq[1][0]:u_seq[1][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] = 3

    return P, S


if __name__ == '__main__':
    # -------------------------------------------------------------------------
    # Generate UserItemData
    #

    np.random.seed(15213)

    true_propensities, true_ratings, user_features, item_features = \
        user_item_features_propensities_ratings(num_users, num_items,
                                                num_components,
                                                np.sqrt(10), -1.5, -0.5)

    for idx in range(10):
        reveal_mask = np.random.binomial(1, true_propensities)
        revealed_ratings = np.zeros((num_users, num_items))
        for u in range(num_users):
            for i in range(num_items):
                if reveal_mask[u, i]:
                    random_rating = \
                        np.round(true_ratings[u, i]
                                 + np.random.normal(scale=1.0))
                    revealed_ratings[u, i] = np.clip(random_rating, 1, 5)

        # give the propensity score naive Bayes approach some true ratings
        # with row/column indices sampled uniformly at random
        mcar_data = (np.random.rand(num_users, num_items) < 0.1) * \
            true_ratings

        trial_dir = os.path.join(useritemfeature_output_dir, str(idx))
        os.makedirs(trial_dir, exist_ok=True)

        np.savetxt(os.path.join(trial_dir, 'user_features.txt'), user_features)
        np.savetxt(os.path.join(trial_dir, 'item_features.txt'), item_features)
        np.savetxt(os.path.join(trial_dir, 'MCAR_data.txt'), mcar_data)

        with open(os.path.join(trial_dir, 'X_observed.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(revealed_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'S.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'P.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_propensities):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))


    # -------------------------------------------------------------------------
    # Generate RowClusterData
    #

    np.random.seed(15213)

    true_propensities, true_ratings = \
        cluster_true_propensities_true_ratings(num_users, num_items,
                                               num_components)

    for idx in range(10):
        reveal_mask = np.random.binomial(1, true_propensities)
        revealed_ratings = np.zeros((num_users, num_items))
        for u in range(num_users):
            for i in range(num_items):
                if reveal_mask[u, i]:
                    random_rating = \
                        np.round(true_ratings[u, i]
                                 + np.random.normal(scale=1.0))
                    revealed_ratings[u, i] = np.clip(random_rating, 1, 5)

        # give the propensity score naive Bayes approach some true ratings
        # with row/column indices sampled uniformly at random
        mcar_data = (np.random.rand(num_users, num_items) < 0.1) * \
            true_ratings

        trial_dir = os.path.join(cluster_output_dir, str(idx))
        os.makedirs(trial_dir, exist_ok=True)

        np.savetxt(os.path.join(trial_dir, 'MCAR_data.txt'), mcar_data)

        with open(os.path.join(trial_dir, 'X_observed.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(revealed_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'S.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'P.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_propensities):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))


    # -------------------------------------------------------------------------
    # Generate MovieLoverData
    #

    np.random.seed(15213)

    # this first step is actually deterministic for MovieLoverData specifically
    true_propensities, true_ratings = \
        steck_propensities_ratings(num_users, num_items)

    for idx in range(10):
        reveal_mask = np.random.binomial(1, true_propensities)
        revealed_ratings = np.zeros((num_users, num_items))
        for u in range(num_users):
            for i in range(num_items):
                if reveal_mask[u, i]:
                    random_rating = \
                        np.round(true_ratings[u, i]
                                 + np.random.normal(scale=1.0))
                    revealed_ratings[u, i] = np.clip(random_rating, 1, 5)

        # give the propensity score naive Bayes approach some true ratings
        # with row/column indices sampled uniformly at random
        mcar_data = (np.random.rand(num_users, num_items) < 0.1) * \
            true_ratings

        trial_dir = os.path.join(steck_output_dir, str(idx))
        os.makedirs(trial_dir, exist_ok=True)

        np.savetxt(os.path.join(trial_dir, 'MCAR_data.txt'), mcar_data)

        with open(os.path.join(trial_dir, 'X_observed.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(revealed_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'S.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_ratings):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
        with open(os.path.join(trial_dir, 'P.txt'), 'w') as f:
            lines = []
            for u, i, r in dense_to_sparse(true_propensities):
                lines.append("%d\t%d\t%f" % (u, i, r))
            f.write("\n".join(lines))
