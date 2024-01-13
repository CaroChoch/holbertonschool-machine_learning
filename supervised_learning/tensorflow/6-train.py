#!/usr/bin/env python3
""" Function that builds, trains, and saves a neural network classifier """
import tensorflow.compat.v1 as tf

# Import necessary modules and functions
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    Function that builds, trains, and saves a neural network classifier

    Arguments:
     - X_train is a numpy.ndarray containing the training input data
     - Y_train is a numpy.ndarray containing the training labels
     - X_valid is a numpy.ndarray containing the validation input data
     - Y_valid is a numpy.ndarray containing the validation labels
     - layer_sizes is a list containing the number of nodes in each layer
        of the network
     - activations is a list containing the activation functions for each
        layer of the network
     - alpha is the learning rate
     - iterations is the number of iterations to train over
     - save_path designates where to save the model

    Returns:
     The path where the model was saved
    """
    # Create placeholders for input data and labels
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Build the computation graph
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    # Add tensors to the TensorFlow graph
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    # Initialize variables and create a Saver object
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Open a TensorFlow session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)

        # Run the training loop
        for i in range(iterations + 1):
            # Calculate loss and accuracy on training data
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_train = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})

            # Calculate loss and accuracy on validation data
            cost_valid = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_valid = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            # Display metrics at regular intervals
            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(cost_train))
                print("\tTraining Accuracy: {}".format(acc_train))
                print("\tValidation Cost: {}".format(cost_valid))
                print("\tValidation Accuracy: {}".format(acc_valid))

            # Execute the training operation to update weights
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        # Save the model once training is complete
        save_path = saver.save(sess, save_path)

    # Return the path where the model was saved
    return save_path
