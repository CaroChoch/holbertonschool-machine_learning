#!/usr/bin/env python3
"""
agglomerative clustering on a dataset
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    performs agglomerative clustering on a dataset
    Arguments:
        - X: np.ndarray (n, d) dataset
        - dist: maximum cophenetic distance
    Returns: clss
        - clss is a np.ndarray (n,) containing the cluster indices for
        each data point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')
    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()
    return clss
