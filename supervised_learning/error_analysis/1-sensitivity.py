#!/usr/bin/env python3
"""
Function that calculates the sensitivity for each class in a confusion
matrix
"""
import numpy as np


def sensitivity(confusion):
    """
    Function that calculates the sensitivity for each class in a confusion
    matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels
        * classes is the number of classes
    Returns:
        a numpy.ndarray of shape (classes,) containing the sensitivity of
        each class
    """
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Negatives (FN) are the sum of each row (actual class) minus TP
    FN = np.sum(confusion, axis=1) - TP

    # Sensitivity for each class is calculated using TP and FN
    sensitivity_per_class = TP / (TP + FN)

    return sensitivity_per_class
