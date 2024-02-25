#!/usr/bin/env python3
"""
Script to perform transfer learning on CIFAR-10 dataset using ResNet152V2
"""

import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data
    Arguments:
        - X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR10 data,
           where m is the number of data points
        - Y: numpy.ndarray of shape (m, ) containing the CIFAR-10 labels for X
        Returns:
            - X_p: numpy.ndarray containing the preprocessed X (data)
            - Y_p: numpy.ndarray containing the preprocessed Y (labels)
    """
    X_p = K.applications.resnet_v2.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == "__main__":
    # Load CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess the data
    X_train, y_train = preprocess_data(X_train, y_train)
    X_test, y_test = preprocess_data(X_test, y_test)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Create a Lambda layer to scale up the data to the correct size
    input_tensor = K.layers.Input(shape=input_shape)
    resize_input = K.layers.Lambda(
        lambda image: K.backend.resize_images(image,
                                              height_factor=(224 // 32),
                                              width_factor=(224 // 32),
                                              data_format="channels_last"))(
                                                  input_tensor)

    # Define ResNet152V2 model
    base_model = K.applications.ResNet152V2(
        include_top=False,
        weights="imagenet",
        input_tensor=resize_input,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=20,
        classifier_activation="softmax",
    )

    # Freeze the layers in the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom head
    x = K.layers.GlobalAveragePooling2D()(base_model.output)
    x = K.layers.Dense(256,
                       activation='relu',
                       kernel_regularizer=K.regularizers.l2(0.001))(x)
    x = K.layers.Dropout(0.3)(x)
    output = K.layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = K.models.Model(inputs=input_tensor, outputs=output)

    # Compile the model
    model.compile(optimizer=K.optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train,
              y_train,
              batch_size=64,
              epochs=5,
              validation_split=0.3)

    # Evaluate the model
    result = model.evaluate(X_test, y_test)
    print("Test loss, Test accuracy:", result)

    # Save the model
    model.save('cifar10.h5')
