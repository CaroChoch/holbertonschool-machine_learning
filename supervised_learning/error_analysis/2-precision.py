#!/usr/bin/env python3
"""
Function that calculates the precision for each class in a confusion matrix
"""
import numpy as np


def precision(confusion):
    """
    Function that calculates the precision for each class in a confusion matrix
    Arguments:
     - confusion is a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
        * classes is the number of classes
        Returns:
         A numpy.ndarray of shape (classes,) containing the precision of
         each class
    """
    # Get the number of classes from the shape of the confusion matrix
    classes = confusion.shape[0]
    # Initialize an array to store precision values for each class
    precision = np.zeros(classes)
    # Calculate precision for each class
    for class_index in range(classes):
        # Precision for the current class is calculated as true positives
        # divided by the sum of predicted positives
        precision[class_index] = confusion[class_index][class_index] / np.sum(
            confusion, axis=0)[class_index]
    # Return the array containing precision values for each class
    return precision
