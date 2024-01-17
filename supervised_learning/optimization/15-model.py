#!/usr/bin/env python3
"""
Function that builds, trains, and saves a neural network model in tensorflow
using Adam optimization, mini-batch gradient descent, learning rate decay,
and batch normalization
"""
import numpy as np
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Function that creates a layer

    Arguments:
        - prev is the tensor output of the previous layer
        - n is the number of nodes in the layer to create
        - activation is the activation function that the layer should use

    Returns:
        The tensor output of the layer
    """
    initializer = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        name="layer")
    return layer(prev)


def create_batch_norm_layer(prev, n, activation):
    """
    Function that creates a batch normalization layer for a neural
    network in tensorflow
    Arguments:
        - prev is the activated output of the previous layer
        - n is the number of nodes in the layer to be created
        - activation is the activation function that should be used
            on the output of the layer
    Returns:
        A tensor of the activated output for the layer
    """
    # Initialize the weights and biases of the layer
    init = tf.keras.initializers.VarianceScaling(mode="fan_avg")
    layer = tf.keras.layers.Dense(units=n,
                                  kernel_initializer=init,
                                  name="layer")

    # Generate the output of the layer
    Z = layer(prev)

    # Calculate the mean and variance of Z
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Gamma and Beta initialization parameters
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(tf.constant(0.0, shape=[n]), name="beta")

    # Epsilon value to avoid division by zero
    epsilon = 1e-8

    # Normalize the output of the layer
    Z_norm = tf.nn.batch_normalization(x=Z,
                                       mean=mean,
                                       variance=variance,
                                       offset=beta,
                                       scale=gamma,
                                       variance_epsilon=epsilon)

    # Return the activation function applied to Z
    if activation is None:
        return Z_norm
    else:
        return activation(Z_norm)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network

    Arguments:
    - x is the placeholder for the input data
    - layer_sizes is a list containing the number of nodes in each layer
        of the network
    - activations is a list containing the activation functions for each
        layer of the network

    Returns:
        The prediction of the network in tensor form
    """
    pred = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            pred = create_batch_norm_layer(
                pred, layer_sizes[i], activations[i])
        else:
            pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred


def shuffle_data(X, Y):
    """
    Function that shuffles the data points in two matrices the same way

    Arguments:
        - X is the first numpy.ndarray of shape (m, nx) to shuffle
            - m is the number of data points
            - nx is the number of features in X
        - Y is the second numpy.ndarray of shape (m, ny) to shuffle
            - m is the same number of data points as in X
            - ny is the number of features in Y
    Returns:
        The shuffled X and Y matrices
    """
    shuffle = np.random.permutation(X.shape[0])
    return X[shuffle], Y[shuffle]


def calculate_loss(y, y_pred):
    """
    Function that calculates the softmax cross-entropy loss of a prediction

    Arguments:
    - y is a placeholder for the labels of the input data
    - y_pred is a tensor containing the network’s predictions

    Returns:
        A tensor containing the loss of the prediction
        """
    loss = tf.compat.v1.losses.softmax_cross_entropy(y, y_pred)
    return loss


def calculate_accuracy(y, y_pred):
    """
    Function that calculates the accuracy of a prediction

    Arguments:
    - y is a placeholder for the labels of the input data
    - y_pred is a tensor containing the network’s predictions

    Returns:
        A tensor containing the decimal accuracy of the prediction
        """
    # Compare the prediction with the labels
    prediction = tf.math.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    # Calculate the accuracy of the prediction and convert tensor bool to float
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    return accuracy


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation in tensorflow
    using inverse time decay
    Arguments:
    - alpha is the original learning rate
    - decay_rate is the weight used to determine the rate at which alpha will
        decay
    - global_step is the number of passes of gradient descent that have
        elapsed
    - decay_step is the number of passes of gradient descent that should occur
        before alpha is decayed further
    Returns:
    The learning rate decay operation
    """
    # Create learning rate decay operation in tf using inverse time decay
    learning_rate_decay = tf.train.inverse_time_decay(learning_rate=alpha,
                                                      global_step=global_step,
                                                      decay_steps=decay_step,
                                                      decay_rate=decay_rate,
                                                      staircase=True)
    return learning_rate_decay


def create_Adam_op(loss, alpha, beta1, beta2, epsilon, global_step):
    """
    Function that creates the training operation for a neural network in
    tensorflow using the Adam optimization algorithm
    Arguments:
    - loss is the loss of the network
    - alpha is the learning rate
    - beta1 is the weight used for the first moment
    - beta2 is the weight used for the second moment
    - epsilon is a small number to avoid division by zero
    Returns:
    The Adam optimization operation
    """

    # Define the optimizer with Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon)

    # Create the Adam optimization operation
    Adam_op = optimizer.minimize(loss, global_step=global_step)

    # Return the Adam optimization operation
    return Adam_op


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a neural network model in
    tensorflow using Adam optimization, mini-batch gradient descent, learning
    rate decay, and batch normalization

    Arguments:
        - Data_train is a tuple containing the training inputs and training
        labels, respectively
        - Data_valid is a tuple containing the validation inputs and validation
        labels, respectively
        - layers is a list containing the number of nodes in each layer of the
        network
        - activations is a list containing the activation functions used for
        each layer of the network
        - alpha is the learning rate
        - beta1 is the weight for the first moment of Adam Optimization
        - beta2 is the weight for the second moment of Adam Optimization
        - epsilon is a small number used to avoid division by zero
        - decay_rate is the decay rate for inverse time decay of the learning
        rate (the corresponding decay step should be 1)
        - batch_size is the number of data points that should be in a
        mini-batch
        - epochs is the number of times the training should pass through the
        whole dataset
        - save_path is the path where the model should be saved to
    Returns:
        The path where the model was saved
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # initialize global_step variable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    decay_step = len(X_train) // batch_size
    if len(X_train) % batch_size != 0:
        decay_step += 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha_decay = learning_rate_decay(alpha, decay_rate,
                                      global_step, decay_step)

    # initizalize train_op and add it to collection
    train_op = create_Adam_op(loss, alpha_decay, beta1,
                              beta2, epsilon, global_step)
    tf.add_to_collection('train_op', train_op)

    # initialize all variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            step = 1
            # print training and validation cost and accuracy
            training_cost = sess.run(
                loss,
                feed_dict={x: X_train, y: Y_train})
            training_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train})
            validation_cost = sess.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid})

            print(f"After {i} epochs:")
            print(f"\tTraining Cost: {training_cost}")
            print(f"\tTraining Accuracy: {training_accuracy}")
            print(f"\tValidation Cost: {validation_cost}")
            print(f"\tValidation Accuracy: {validation_accuracy}")

            # shuffle data
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch from X_train shuffled and
                # Y_train shuffled
                X_batch = shuffled_X[j:j + batch_size, :]
                Y_batch = shuffled_Y[j:j + batch_size, :]

                # run training operation
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # print batch cost and accuracy
                if step > 0 and step % 100 == 0:
                    batch_cost = sess.run(
                        loss,
                        feed_dict={x: X_batch, y: Y_batch})
                    batch_accuracy = sess.run(
                        accuracy,
                        feed_dict={x: X_batch, y: Y_batch})
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {batch_cost}")
                    print(f"\t\tAccuracy: {batch_accuracy}")
                step += 1

        # print training and validation cost and accuracy again
        training_cost = sess.run(
            loss,
            feed_dict={x: X_train, y: Y_train})
        training_accuracy = sess.run(
            accuracy,
            feed_dict={x: X_train, y: Y_train})
        validation_cost = sess.run(
            loss,
            feed_dict={x: X_valid, y: Y_valid})
        validation_accuracy = sess.run(
            accuracy,
            feed_dict={x: X_valid, y: Y_valid})

        print(f"After {epochs} epochs:")
        print(f"\tTraining Cost: {training_cost}")
        print(f"\tTraining Accuracy: {training_accuracy}")
        print(f"\tValidation Cost: {validation_cost}")
        print(f"\tValidation Accuracy: {validation_accuracy}")

        # save and return the path to where the model was saved
        save_path = saver.save(sess, save_path)
    return save_path
