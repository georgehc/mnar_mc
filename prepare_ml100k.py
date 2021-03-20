"""
Splits the MovieLens 100K randomly 10 times into 90/10 train/test splits for
use with our demo.

This script expects MovieLens 100K data to be in the folder './ml-100k'. In
particular, the ratings data should be in './ml-100k/u.data'.

Data for the 10 splits are Surprise-compatible and are saved in the folders:
    ml-100k/0/
    ml-100k/1/
    ml-100k/2/
    ...
    ml-100k/9/

Per folder, we output "X_observed.txt" (train) and "X_test.txt" (test).

Authors: George H. Chen (georgechen@cmu.edu), Wei Ma (wei.w.ma@polyu.edu.hk)
"""
import csv
import numpy as np
import os


movielens_data_filename = 'ml-100k/u.data'
output_dir = 'ml-100k'

raw_ratings = []
with open(movielens_data_filename, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 3:
            # assume format of <user> <item> <rating>
            raw_ratings.append((int(row[0]), int(row[1]), float(row[2])))

def unique_entries_no_sort(x):
    """
    Given a numpy array, outputs its unique values *not* sorted (i.e., in order
    of appearance)

    This snippet of code is from here:
    https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    """
    indices = np.unique(x, return_index=True)[1]
    return np.array([x[idx] for idx in sorted(indices)])

num_ratings = len(raw_ratings)
raw_ratings = np.array(raw_ratings)
unique_users = unique_entries_no_sort(raw_ratings[:, 0].astype(np.int))
unique_items = unique_entries_no_sort(raw_ratings[:, 1].astype(np.int))
num_users = unique_users.max()  # note that MovieLens indexes from 1
num_items = unique_items.max()

# Surprise reindexes the rows/columns by order of appearance; we also reindex
row_raw_to_new = {raw_u: inner_u for inner_u, raw_u in enumerate(unique_users)}
col_raw_to_new = {raw_i: inner_i for inner_i, raw_i in enumerate(unique_items)}
reindexed_ratings = np.array([(row_raw_to_new[int(raw_u)],
                               col_raw_to_new[int(raw_i)], r)
                              for raw_u, raw_i, r in raw_ratings])

def dense_to_sparse(X):
    m, n = X.shape
    return [(u, i, X[u, i])
            for u in range(m)
            for i in range(n)
            if X[u, i] != 0]

for idx in range(10):
    np.random.seed(idx)
    test_indices = np.random.choice(num_ratings,
                                    size=int(.1*num_ratings),
                                    replace=False)
    test_mask = np.zeros(num_ratings, dtype=np.bool)
    test_mask[test_indices] = 1
    train_mask = ~test_mask

    os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)

    train_filename = os.path.join(output_dir, str(idx), 'X_observed.txt')
    train_matrix = np.zeros((num_users, num_items))
    for u, i, r in reindexed_ratings[train_mask]:
        train_matrix[int(u), int(i)] = r
    with open(train_filename, 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(train_matrix):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))

    test_filename = os.path.join(output_dir, str(idx), 'X_test.txt')
    test_matrix = np.zeros((num_users, num_items))
    for u, i, r in reindexed_ratings[test_mask]:
        test_matrix[int(u), int(i)] = r
    with open(test_filename, 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(test_matrix):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
