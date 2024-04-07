#!/usr/bin/env python3
"""
Performs K-means on a dataset
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset
    Arguments:
        - X: np.ndarray (n, d) data set
            - n: number of data points
            - d: number of dimensions
        - k: positive int, number of clusters
    Returns: C, clss
        - C is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
        - clss is a numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to
    """
    # Create the KMeans object
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    # Get the cluster centers and labels
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
