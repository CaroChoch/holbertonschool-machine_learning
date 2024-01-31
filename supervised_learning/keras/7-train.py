#!/usr/bin/env python3
""" Function that trains a model using learrning rate decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """
    Function that trains a model using learrning rate decay
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
        - learning_rate_decay: is a boolean that indicates whether
        learning rate decay should be used
        - alpha: is the initial learning rate
        - decay_rate: is the decay rate
        - verbose: is a boolean that determines if output should be
        printed during training
        - shuffle: is a boolean that determines whether to shuffle the
        batches every epoch. Defaults to False.
    Returns: the History object generated after training the model
    """
    def learning_rate_schedule(epoch):
        """ Updates the learning rate using inverse time decay """
        return alpha / (1 + decay_rate * epoch)

    callback = []

    # Check if early stopping is requested and validation data is provided
    if early_stopping and validation_data:
        callback.append(K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience))

    # Check if learning rate decay is requested and validation data is provided
    if learning_rate_decay and validation_data:
        callback.append(K.callbacks.LearningRateScheduler(
            schedule=learning_rate_schedule,
            verbose=True))

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
