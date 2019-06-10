#!/usr/bin/python3
import numpy as np

"""
X - observation matrix:
    rows - observations; n - number of rows
    cols - values/factors observed
    
@ operator - matrix multiplication in numpy
"""

# np.loadtxt(open("cereal_cut.csv", "rb"), delimiter=",")

def center_matrix(X):
    """
    centers a matrix by subtracting matrix A with values 1/n from it
    :param X: :type numpy array: input matrix
    :return: :type numpy array:  - centered matrix
    """
    n = len(X)
    A = np.divide(np.ones((n, n)), n)
    M = A @ X

    return X - M


def reduce(X, threshold = 1, leave = None):
    """
    reduces the dimensionality of a observed data by removing the least significant characteristics
    :param X: :type numpy array: input matrix
    :param threshold: :type float: cutoff value for eigenvalues to be considered significant
    :param leave: :type int: fallback if nothing is cut off, leave a fixed number of the most significant axes, None - don't cut off anything
    :return: :type numpy array: reduced matrix
    """
    n = len(X)
    X_hat = center_matrix(X)

    # Sigma
    cov_mtx = np.cov(X_hat, rowvar = False) # (X_hat.transpose() @ X_hat) * (1 / n)  -- difrerent results ???

    cov_eVal, cov_eVec = np.linalg.eigh(cov_mtx)

    cutoff_index = 0
    for ind,eVal in enumerate(cov_eVal):
        if eVal > threshold:
            cutoff_index = ind
            break

    if cutoff_index == 0 and leave:
        cutoff_index = len(eVal) - leave if len(eVal) - leave >= 0 else 0

    return X_hat @ cov_eVec[ : , cutoff_index : ]

if __name__ == "__main__":
    d = np.loadtxt(open("./data/cereal_cut.csv", "rb"), delimiter=",", skiprows=1)
    print(d.shape, d)
    res = reduce(d)
    print(res.shape, res)
