#!/usr/bin/env python3
"""
Function that trains a loaded neural network model using mini-batch gradient
descent
"""
import tensorflow.compat.v1 as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Function that trains a loaded neural network model using
    mini-batch gradient descent

    Arguments:
        - X_train is a numpy.ndarray of shape (m, 784) containing the training
            data
            * m is the number of data points
            * 784 is the number of input features
        - Y_train is a one-hot numpy.ndarray of shape (m, 10) containing the
            training labels
            * 10 is the number of classes the model should classify
        - X_valid is a numpy.ndarray of shape (m, 784) containing the
            validation data
        - Y_valid is a one-hot numpy.ndarray of shape (m, 10) containing the
            validation labels
        - batch_size is the number of data points in a batch
        - epochs is the number of times the training should pass through the
            whole dataset
        - load_path is the path from which to load the model
        - save_path is the path to where the model should be saved after
            training
    Returns:
        The path where the model was saved
    """
    with tf.Session() as session:
        # Restore the saved model
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(session, load_path)

        # Get the following tensors and ops from the restored collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # Loop over the number of epochs
        for epoch in range(epochs + 1):
            # Evaluate the training cost and accuracy of the model
            training_cost = session.run(loss,
                                        feed_dict={x: X_train, y: Y_train})
            training_accuracy = session.run(accuracy,
                                            feed_dict={x: X_train,
                                                       y: Y_train})
            validation_cost = session.run(loss,
                                          feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = session.run(accuracy,
                                              feed_dict={x: X_valid,
                                                         y: Y_valid})

            print(f"After {epoch} epochs:")
            print(f"\tTraining Cost: {training_cost}")
            print(f"\tTraining Accuracy: {training_accuracy}")
            print(f"\tValidation Cost: {validation_cost}")
            print(f"\tValidation Accuracy: {validation_accuracy}")

            # Save the model after the last epoch
            if epoch == epochs:
                return saver.save(session, save_path)

            # shuffle the training data for each epoch
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            # Generate indices for mini-batch
            steps = list(range(0, len(shuffled_X), batch_size))

            # Loop over mini-batch
            for step, start_index in enumerate(steps, start=1):
                end_index = start_index + batch_size
                X_batch = shuffled_X[start_index:end_index]
                Y_batch = shuffled_Y[start_index:end_index]
                session.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # Print progress every 100 steps
                if step % 100 == 0 and step != 1:
                    step_cost = session.run(loss, feed_dict={x: X_batch,
                                                             y: Y_batch})
                    step_accuracy = session.run(accuracy,
                                                feed_dict={x: X_batch,
                                                           y: Y_batch})
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")
        # save the trained session
        save_path = saver.save(session, save_path)
    return save_path
