#!/usr/bin/env python3
""" Function that determines if you should stop gradient descent early """


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Function that determines if you should stop gradient descent early
    Arguments:
     - cost is the current validation cost of the neural network
     - opt_cost is the lowest recorded validation cost of the neural network
     - threshold is the threshold used for early stopping
     - patience is the patience count used for early stopping
     - count is the count of how long the threshold has not been met
     Returns:
      a boolean of whether the network should be stopped early, followed
      by the updated count
    """
    early_stopping = True
     # Check if the current cost improvement is greater than the threshold
    if (opt_cost - cost) > threshold:
        # Reset the count as there is improvement
        count = 0
    else:
        # Increment the count as there is no significant improvement
        count += 1
    # Check if the count has not reached the patience limit
    if count != patience:
        # Continue training, no early stopping
        early_stopping = False

    # Stop training, early stopping
    return early_stopping, count
