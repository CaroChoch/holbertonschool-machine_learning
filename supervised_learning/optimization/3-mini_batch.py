#!/usr/bin/env python3
"""Create mini-batches for mini-batch gradient descent"""

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size=32):
    """
    Creates mini-batches from (X, Y)

    Args:
        X: numpy.ndarray of shape (m, nx)
        Y: numpy.ndarray of shape (m, ny)
        batch_size: size of each batch

    Returns:
        List of tuples (X_batch, Y_batch)
    """
    m = X.shape[0]
    X, Y = shuffle_data(X, Y)

    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
