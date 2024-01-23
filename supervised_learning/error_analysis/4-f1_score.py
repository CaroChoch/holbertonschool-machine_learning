#!/usr/bin/env python3
""" Function taht calculates the F1 score of a confusion matrix """
import numpy as np


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
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Positives(FP) are the sum of each column (predicted class) minus TP
    FP = np.sum(confusion, axis=0) - TP

    # False Negatives (FN) are the sum of each row (actual class) minus TP
    FN = np.sum(confusion, axis=1) - TP

    # Precision for each class is calculated using TP and FP
    precision = TP / (TP + FP)

    # Recall for each class is calculated using TP and FN
    recall = TP / (TP + FN)

    # F1 score for each class is calculated using precision and recall
    F1_score = 2 * precision * recall / (precision + recall)

    # Return the array containing F1 score values for each class
    return F1_score
