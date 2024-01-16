#!/usr/bin/env python3
""" Function that calculates the weighted moving average of a data set """


def moving_average(data, beta):
    """
    Function that calculates the weighted moving average of a data set
    Arguments:
     - data is the list of data to calculate the moving average of
     - beta is the weight used for the moving average
    Returns:
     A list containing the moving averages of data
    """
    # Initialize the weighted cumulative sum to zero
    weighted_sum = 0
    # Initialize the bias correction to zero
    bias_correction = 0
    # List to store the weighted moving averages of data
    weight_moving_average_values = []

    # Loop through each data point in the list
    for i in range(len(data)):
        # Update the weighted cumulative sum
        weighted_sum = beta * weighted_sum + (1 - beta) * data[i]
        # Update the bias correction (cumulative sum of weighting coefficients)
        bias_correction = bias_correction * beta + (1 - beta)
        # Calculate the weighted moving average with bias correction
        weight_moving_average = weighted_sum / bias_correction
        # Append the weighted moving average to the list
        weight_moving_average_values.append(weight_moving_average)

    # return the list of weighted moving averages
    return weight_moving_average_values
