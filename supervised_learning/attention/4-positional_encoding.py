#!/usr/bin/env python3
""" Positional Encoding """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer
    Arguments:
        - max_seq_len is an integer representing the maximum sequence length
        - dm is the model depth
    Returns: a numpy.ndarray of shape (max_seq_len, dm) containing
        the positional encoding vectors
    """
    # Initialize the positional encoding matrix with zeros
    PE = np.zeros((max_seq_len, dm))

    # Loop over each position in the sequence
    for i in range(max_seq_len):
        # Loop over each dimension of the positional encoding
        for j in range(0, dm, 2):
            # Compute the positional encoding using sin (even indices)
            PE[i, j] = np.sin(i / (10000 ** (j / dm)))
            # Compute the positional encoding using cos (odd indices)
            PE[i, j + 1] = np.cos(i / (10000 ** (j / dm)))

    # Return the positional encoding matrix
    return PE
