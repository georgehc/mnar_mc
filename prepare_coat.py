"""
Converts the Coat dataset (Schnabel et al 2016) to be Surprise-compatible for
use with our demo.

This script expects Coat data to be in the folder './coat'. It converts the
Coat train/test datasets to be in the expected Surprise format.

Author: George H. Chen (georgechen@cmu.edu)

Reference:

    Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and
    Thorsten Joachims.Recommendations as treatments: Debiasing learning and
    evaluation. In International Conferenceon Machine Learning, pages
    1670–1679, 2016
"""
import numpy as np


def dense_to_sparse(X):
    m, n = X.shape
    return [(u, i, X[u, i])
            for u in range(m)
            for i in range(n)
            if X[u, i] != 0]

if __name__ == '__main__':
    with open('coat/train_surprise_format.csv', 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(np.loadtxt('coat/train.ascii')):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))

    with open('coat/test_surprise_format.csv', 'w') as f:
        lines = []
        for u, i, r in dense_to_sparse(np.loadtxt('coat/test.ascii')):
            lines.append("%d\t%d\t%f" % (u, i, r))
        f.write("\n".join(lines))
