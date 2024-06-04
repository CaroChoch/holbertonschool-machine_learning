#!/usr/bin/env python3
""" Generating faces """
import tensorflow as tf
from tensorflow import keras


# load the pictures
def convolutional_GenDiscr():
    """
    Create a generator and a discriminator
    """
    def get_generator():
        """
        Create a generator network
        """
        input_1 = keras.layers.Input(shape=(16,))
        dense = keras.layers.Dense(
            2048, activation=keras.layers.Activation("tanh"))(input_1)
        reshape = keras.layers.Reshape((2, 2, 512))(dense)

        up_sampling2d = keras.layers.UpSampling2D((2, 2))(reshape)
        conv2d = keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding="same")(up_sampling2d)
        batch_normalization = keras.layers.BatchNormalization()(conv2d)
        activation_1 = keras.layers.Activation("tanh")(batch_normalization)

        up_sampling2d_1 = keras.layers.UpSampling2D((2, 2))(activation_1)
        conv2d_1 = keras.layers.Conv2D(
            16, (3, 3), strides=(1, 1), padding="same")(up_sampling2d_1)
        batch_normalization_1 = keras.layers.BatchNormalization()(conv2d_1)
        activation_2 = keras.layers.Activation("tanh")(batch_normalization_1)

        up_sampling2d_2 = keras.layers.UpSampling2D((2, 2))(activation_2)
        conv2d_2 = keras.layers.Conv2D(
            1, (3, 3), strides=(1, 1), padding="same")(up_sampling2d_2)
        batch_normalization_2 = keras.layers.BatchNormalization()(conv2d_2)

        activation_3 = keras.layers.Activation("tanh")(batch_normalization_2)

        return keras.models.Model(input_1, activation_3, name="generator")

    def get_discriminator():
        """
        Create a discriminator network
        """
        input_2 = keras.layers.Input(shape=(16, 16, 1))

        conv2d_3 = keras.layers.Conv2D(
            32, (3, 3), strides=(1, 1), padding="same")(input_2)
        max_pooling2d = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_3)
        activation_4 = keras.layers.Activation("tanh")(max_pooling2d)
        conv2d_4 = keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding="same")(activation_4)
        max_pooling2d_1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_4)
        activation_5 = keras.layers.Activation("tanh")(max_pooling2d_1)
        conv2d_5 = keras.layers.Conv2D(
            128, (3, 3), strides=(1, 1), padding="same")(activation_5)
        max_pooling2d_2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_5)
        activation_6 = keras.layers.Activation("tanh")(max_pooling2d_2)
        conv2d_6 = keras.layers.Conv2D(
            256, (3, 3), strides=(1, 1), padding="same")(activation_6)
        max_pooling2d_3 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_6)
        activation_7 = keras.layers.Activation("tanh")(max_pooling2d_3)
        flatten = keras.layers.Flatten()(activation_7)
        dense_1 = keras.layers.Dense(1, activation="tanh")(flatten)
        return keras.models.Model(input_2, dense_1, name="discriminator")

    return get_generator(), get_discriminator()
