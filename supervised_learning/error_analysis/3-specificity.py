#!/usr/bin/env python3
"""
Function that calculates the specificity for each class in a confusion matrix
"""
import numpy as np


def specificity(confusion):
    """
    Function that calculates the specificity for each class in a confusion
    matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes) where
        row indices represent the correct labels and column indices represent
        the predicted labels
        * classes is the number of classes
    Returns:
        A numpy.ndarray of shape (classes,) containing the specificity of
        each class
    """
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Positives(FP) are the sum of each column (predicted class) minus TP
    FP = np.sum(confusion, axis=0) - TP

    # False Negatives (FN) are the sum of each row (actual class) minus TP
    FN = np.sum(confusion, axis=1) - TP

    # True Negatives (TN) are calculated by subtracting TP, FP, and FN from
    # the total number of elements in the matrix
    TN = np.sum(confusion) - (FP + FN + TP)

    # True Negative Rate (TNR or Specificity) for each class is calculated
    # using TN and FP
    TNR = TN / (TN + FP)

    # Return the array containing specificity values for each class
    return TNR
