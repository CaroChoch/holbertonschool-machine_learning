#!/usr/bin/env python3
""" Function that evaluates the output of a neural network """
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Function that evaluates the output of a neural network

    Arguments:
     - X is a numpy.ndarray containing the input data to evaluate
     - Y is a numpy.ndarray containing the one-hot labels for X
     - save_path is the location to load the model from

    Returns:
     The network’s prediction, accuracy, and loss, respectively
    """
    # Create a TensorFlow session
    with tf.Session() as sess:
        # Import the saved model's graph
        saver = tf.train.import_meta_graph(save_path + ".meta")
        # Restore the model's variables from the saved checkpoint
        saver.restore(sess, save_path)

        # Retrieve input placeholder, output tensor, loss, and accuracy
        # from the saved graph
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        loss = tf.get_collection("loss")[0]
        accuracy = tf.get_collection("accuracy")[0]

        # Run prediction, loss, and accuracy operations on the provided data
        y_pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        loss = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy = sess.run(accuracy, feed_dict={x: X, y: Y})

        # Return the network's prediction, accuracy, and loss
        return y_pred, accuracy, loss
