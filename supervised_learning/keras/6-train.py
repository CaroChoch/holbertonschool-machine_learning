#!/usr/bin/env python3
""" Function that trains a model using early stopping"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """
    Function that trains a model using early stopping
    Arguments:
        - network: is the model to train
        - data: is a numpy.ndarray of shape (m, nx) containing
        the input data
        - labels: is a one-hot numpy.ndarray of shape (m, classes)
        containing the labels of data
        - batch_size: is the size of the batch used for mini-batch
        gradient descent
        - epochs: is the number of passes through data for mini-batch
        gradient descent
        - validation_data : is the data to validate the model with
        - early_stopping: is a boolean that indicates whether early
        stopping should be used
        - patience: is the patience used for early stopping
        - verbose: is a boolean that determines if output should be
        printed during training
        - shuffle: is a boolean that determines whether to shuffle the
        batches every epoch. Defaults to False.
    Returns: the History object generated after training the model
    """

    callback = []

    # Create EarlyStopping callback if early_stopping is enabled
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience))
    # Train the model using fit method
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          callbacks=callback,
                          shuffle=shuffle)

    # Return the training history
    return history
