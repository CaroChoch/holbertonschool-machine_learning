#!/usr/bin/env python3
"""
Function that builds a modified version of the LeNet-5 architecture
using tensorflow
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    Function that builds a modified version of the LeNet-5 architecture
    using tensorflow
    Arguments:
        - x: is a tf.placeholder of shape (m, 28, 28, 1) containing the
        input images for the network
            * m is the number of images
        - y: is a tf.placeholder of shape (m, 10) containing the one-hot
        labels for the network
    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default
        hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Layer 1: Convolutional layer with 6 kernels of shape 5x5 and same padding
    convolutional_layer1 = tf.layers.Conv2D(filters=6,
                                            kernel_size=(5, 5),
                                            padding='same',
                                            activation='relu',
                                            kernel_initializer=initializer)(x)

    # Layer 2: Max pooling layer with kernels of shape 2x2 and 2x2 strides
    max_pool_layer1 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2))(
                                                 convolutional_layer1)

    # Layer 3: Convolutional layer with 16 kernels of shape 5x5 and valid pad
    convolutional_layer2 = tf.layers.Conv2D(filters=16,
                                            kernel_size=(5, 5),
                                            padding='valid',
                                            activation='relu',
                                            kernel_initializer=initializer)(
                                                max_pool_layer1)

    # Layer 4: Max pooling layer with kernels of shape 2x2 and 2x2 strides
    max_pool_layer2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                             strides=(2, 2))(
                                                 convolutional_layer2)

    # Flatten the output for fully connected layers
    flattened = tf.layers.Flatten()(max_pool_layer2)

    # Layer 5: Fully connected layer with 120 nodes
    fully_connected_layer1 = tf.layers.Dense(units=120,
                                             activation='relu',
                                             kernel_initializer=initializer)(
                                                 flattened)

    # Layer 6: Fully connected layer with 84 nodes
    fully_connected_layer2 = tf.layers.Dense(units=84,
                                             activation='relu',
                                             kernel_initializer=initializer)(
                                                 fully_connected_layer1)

    # Layer 7: Fully connected softmax output layer with 10 nodes
    output_layer = tf.layers.Dense(units=10,
                                   activation=None,
                                   kernel_initializer=initializer)(
                                       fully_connected_layer2)

    # GET TENSORS FOR THE RETURN
    # Tensor for the softmax activated output
    softmax_tensor = tf.nn.softmax(output_layer)

    # Define the loss function
    loss_tensor = tf.losses.softmax_cross_entropy(onehot_labels=y,
                                                  logits=output_layer)

    # Define the training operation using Adam optimizer
    optimizer = tf.train.AdamOptimizer()
    training_operation = optimizer.minimize(loss_tensor)

    # Define the accuracy metric
    correct_predictions = tf.equal(tf.argmax(output_layer, axis=1),
                                   tf.argmax(y, axis=1))
    accuracy_tensor = tf.reduce_mean(tf.cast(correct_predictions,
                                             dtype=tf.float32))

    return softmax_tensor, training_operation, loss_tensor, accuracy_tensor
