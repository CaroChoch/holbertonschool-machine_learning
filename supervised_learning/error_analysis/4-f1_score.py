#!/usr/bin/env python3
""" Function taht calculates the F1 score of a confusion matrix """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Function taht calculates the F1 score of a confusion matrix
    Arguments:
        - confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels and column indices
            represent the predicted labels
            * classes is the number of classes
    Returns:
        A numpy.ndarray of shape (classes,) containing the F1 score of each
        class
    """
    # Initialize an array to store F1 score values for each class
    F1_score = np.zeros(confusion.shape[0])
    # Calculate sensitivity and precision using the provided functions
    sensitivity_values = sensitivity(confusion)
    precision_values = precision(confusion)

    # Calculate F1 score for each class using sensitivity and precision
    F1_score = 2 * (sensitivity_values * precision_values) / (
        sensitivity_values + precision_values)

    return F1_score
