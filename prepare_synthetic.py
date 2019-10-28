import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegressionCV
from mc_algorithms import std_logistic_function


num_components = 20
num_users = 200
num_items = 300

steck_output_dir = 'steck'
useritemfeature_output_dir = 'useritemfeature'


def dense_to_sparse(X):
    m, n = X.shape
    return [(u, i, X[u, i])
            for u in range(m)
            for i in range(n)
            if X[u, i] != 0]


def steck_true_propensities_true_ratings(num_rows, num_cols, p=0.5):
    u_seq = np.array_split(range(num_rows), 2)
    i_seq = np.array_split(range(num_cols), 3)

    true_P = np.ones((num_rows, num_cols)) * p
    true_P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] *= 1
    true_P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] *= 0.1
    true_P[u_seq[0][0]:u_seq[0][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] *= 0.5
    true_P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] *= 0.1
    true_P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] *= 1
    true_P[u_seq[1][0]:u_seq[1][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] *= 0.5

    true_X = np.zeros((num_rows, num_cols))
    true_X[u_seq[0][0]:u_seq[0][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] = 5
    true_X[u_seq[0][0]:u_seq[0][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] = 1
    true_X[u_seq[0][0]:u_seq[0][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] = 3
    true_X[u_seq[1][0]:u_seq[1][-1]+1, i_seq[0][0]:i_seq[0][-1]+1] = 1
    true_X[u_seq[1][0]:u_seq[1][-1]+1, i_seq[1][0]:i_seq[1][-1]+1] = 5
    true_X[u_seq[1][0]:u_seq[1][-1]+1, i_seq[2][0]:i_seq[2][-1]+1] = 3

    return true_P, true_X


def user_item_features_true_propensities_true_ratings(num_rows, num_cols,
                                                      num_components):
    true_U = np.random.rand(num_rows, num_components)
    true_V = np.random.rand(num_cols, num_components)
    true_X = np.dot(true_U, true_V.T)
    true_X = (true_X - true_X.min()) / (true_X.max() - true_X.min())
    true_X = 1 + np.round(true_X * 4)
    true_U_feature = np.random.randn(num_rows, num_components) / 8
    true_V_feature = np.random.randn(num_cols, num_components) / 8

    param_U = np.random.rand(num_components)
    param_V = np.random.rand(num_components)

    true_P = np.ones((num_rows, num_cols))
    for u in range(num_rows):
        for i in range(num_cols):
            true_P[u, i] = \
                std_logistic_function(np.inner(true_U_feature[u], param_U) +
                                      np.inner(true_V_feature[i], param_V))

    return true_P, true_X, true_U_feature, true_V_feature


def estimate_propensities_naive_bayes_MAR(x_train, x_mar, min_prob=0.05):
    """
    The propensity score estimation strategy given by equation (18) of
    Schnabel et al 2016.

    Reference:

        Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and
        Thorsten Joachims.Recommendations as treatments: Debiasing learning and
        evaluation. In International Conferenceon Machine Learning, pages
        1670–1679, 2016
    """
    if type(x_train) != csr_matrix:
        x_train = csr_matrix(x_train)
    if type(x_mar) != csr_matrix:
        x_mar = csr_matrix(x_mar)

    r_array = np.unique(x_train.data)
    num_train_data = len(x_train.data)

    p_o = num_train_data / np.float(x_train.shape[0] * x_train.shape[1])

    x_total = x_train + x_mar
    num_total_data = len(x_total.data)

    r_prob_dict = dict()
    for r in r_array:
        r_prob_dict[r] = ((x_train.data == r).sum() / num_train_data, 
                          (x_total.data == r).sum() / num_total_data)

    P_hat = np.zeros(x_train.shape)
    for i in range(x_train.shape[0]):
        for idx, j in enumerate(x_train.indices[x_train.indptr[i]:
                                                x_train.indptr[i+1]]):
            pro, pr = r_prob_dict[x_train.data[x_train.indptr[i]:
                                               x_train.indptr[i+1]][idx]]
            if pr > 0:
                P_hat[i, j] = pro * p_o / pr

    # weird edge cases that shouldn't matter
    inf_mask = np.isinf(P_hat)
    if inf_mask.sum() > 0:
        print('*** WARNING: Infinity in propensity score estimation')
        P_hat[inf_mask] = np.nan
    zero_mask = P_hat == 0
    if zero_mask.sum() > 0:
        P_hat[zero_mask] = np.nan
    nan_mask = np.isnan(P_hat)
    if nan_mask.sum() > 0:
        P_hat[nan_mask] = np.nanmean(P_hat)
    P_hat = np.maximum(P_hat, min_prob)
    return P_hat


def estimate_propensities_log_reg(revealed_ratings,
                                  user_features,
                                  item_features,
                                  min_prob=0.05):
    num_users, num_items = revealed_ratings.shape
    feature_vectors = []
    labels = []
    for u in range(num_users):
        for i in range(num_items):
            feature_vector = np.concatenate(([1],
                                             user_features[u],
                                             item_features[i]))
            feature_vectors.append(feature_vector)
            if revealed_ratings[u, i] > 0:
                labels.append(1)
            else:
                labels.append(0)
    feature_vectors = np.array(feature_vectors)
    labels = np.array(labels)
    clf = LogisticRegressionCV(Cs=5, cv=5, scoring='neg_log_loss',
                               solver='lbfgs')
    clf.fit(feature_vectors, labels)
    predictions = clf.predict_proba(feature_vectors)[:, 1]
    P_hat = np.maximum(predictions.reshape((num_users, num_items)), min_prob)
    return P_hat


# -----------------------------------------------------------------------------
# Generate MovieLoverData
#

true_propensities, true_ratings = \
    steck_true_propensities_true_ratings(num_users, num_items)  # deterministic

np.random.seed(15213)

for idx in range(10):
    reveal_mask = np.random.binomial(1, true_propensities)
    revealed_ratings = np.zeros((num_users, num_items))
    for u in range(num_users):
        for i in range(num_items):
            if reveal_mask[u, i]:
                random_rating = \
                    np.round(true_ratings[u, i] + np.random.normal(scale=1.0))
                revealed_ratings[u, i] = np.clip(random_rating, 1, 5)

    # give the propensity score naive Bayes approach some true ratings
    # with row/column indices sampled uniformly at random
    mask = np.random.rand(num_users, num_items) < 0.1
    P_NB = estimate_propensities_naive_bayes_MAR(revealed_ratings,
                                                 mask * true_ratings)

    trial_dir = os.path.join(steck_output_dir, str(idx))
    os.makedirs(trial_dir, exist_ok=True)

    with open(os.path.join(trial_dir, 'observed_X.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(revealed_ratings):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'P_NB.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(P_NB):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'whole_X.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(true_ratings):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'true_P.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(true_propensities):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))


# -----------------------------------------------------------------------------
# Generate UserItemData
#

np.random.seed(15213)

true_propensities, true_ratings, user_features, item_features = \
    user_item_features_true_propensities_true_ratings(num_users, num_items,
                                                      num_components)  # random

for idx in range(10):
    reveal_mask = np.random.binomial(1, true_propensities)
    revealed_ratings = np.zeros((num_users, num_items))
    for u in range(num_users):
        for i in range(num_items):
            if reveal_mask[u, i]:
                random_rating = \
                    np.round(true_ratings[u, i] + np.random.normal(scale=1.0))
                revealed_ratings[u, i] = np.clip(random_rating, 1, 5)

    # give the propensity score naive Bayes approach some true ratings
    # with row/column indices sampled uniformly at random
    mask = np.random.rand(num_users, num_items) < 0.1
    P_NB = estimate_propensities_naive_bayes_MAR(revealed_ratings,
                                                 mask * true_ratings)

    P_LR = estimate_propensities_log_reg(revealed_ratings,
                                         user_features,
                                         item_features)

    trial_dir = os.path.join(useritemfeature_output_dir, str(idx))
    os.makedirs(trial_dir, exist_ok=True)

    np.savetxt(os.path.join(trial_dir, 'user_features.txt'), user_features)
    np.savetxt(os.path.join(trial_dir, 'item_features.txt'), item_features)

    with open(os.path.join(trial_dir, 'observed_X.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(revealed_ratings):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'P_LR.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(P_LR):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'P_NB.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(P_NB):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'whole_X.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(true_ratings):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
    with open(os.path.join(trial_dir, 'true_P.txt'), 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(true_propensities):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
