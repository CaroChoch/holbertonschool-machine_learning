#!/usr/bin/env python3
"""
calculates a GMM from a dataset
"""
import sklearn.mixture


def gmm(X, k):
    """
    calculates a GMM from a dataset
    Arguments:
        - X: np.ndarray, (n, d), dataset
        - k: positive int, number of clusters
    Returns: pi, m, S, clss, bic
        - pi: np.ndarray, (k,), priors for each cluster
        - m: np.ndarray, (k, d), centroid means for each cluster
        - S: np.ndarray, (k, d, d), covariance matrices for each cluster
        - clss: np.ndarray of shape (n,) containing the cluster indices for each data point
        - bic: np.ndarray of shape (kmax - kmin + 1) containing the BIC value for each cluster size tested
    """
    # Create the GMM object
    gaussian_mixture = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    # Get the cluster centers and labels
    pi = gaussian_mixture.weights_
    m = gaussian_mixture.means_
    S = gaussian_mixture.covariances_
    clss = gaussian_mixture.predict(X)
    bic = gaussian_mixture.bic(X)

    return pi, m, S, clss, bic
