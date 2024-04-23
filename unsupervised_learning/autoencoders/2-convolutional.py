#!/usr/bin/env python3
""" Convolutional Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder
    Arguments:
        - input_dims: integer containing dimensions of model input
        - filters: list containing number of filters for each convolutional
            layer in the encoder, respectively
        - latent_dims: integer containing dimensions of latent space
    Returns: encoder, decoder, auto
        - encoder: the encoder model
        - decoder: the decoder model
        - auto: the full autoencoder model
    """
    # ENCODER
    # Input layer
    input_layer = keras.Input(shape=input_dims)
    # Convolutional layers
    enc_conv = input_layer
    for filters_num in filters:
        enc_conv = keras.layers.Conv2D(
            filters=filters_num,
            kernel_size=(3, 3),
            activation='relu',
            padding='same')(enc_conv)
        enc_conv = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same')(enc_conv)
    encoder_output = enc_conv

    # Create encoder model
    encoder_model = keras.models.Model(input_layer, encoder_output)

    # DECODER
    # Input layer
    latent_input = keras.Input(shape=(latent_dims))
    # Hidden layers
    dec_hidden = latent_input

    # Convolutional layers
    for i in reversed(filters[1:]):
        dec_hidden = keras.layers.Conv2D(
            filters=i,
            kernel_size=(3, 3),
            activation='relu',
            strides=(1, 1),
            padding='same')(dec_hidden)
        dec_hidden = keras.layers.UpSampling2D((2, 2))(dec_hidden)
    # Last convolutional layer
    dec_hidden = keras.layers.Conv2D(
        filters=filters[-1],
        kernel_size=(3, 3),
        activation='relu',
        strides=(1, 1),
        padding='valid')(dec_hidden)
    dec_hidden = keras.layers.UpSampling2D((2, 2))(dec_hidden)
    # Decoder output
    dec_output = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        strides=(1, 1),
        activation='sigmoid',
        padding='same')(dec_hidden)
    # Create decoder model
    decoder_model = keras.models.Model(
        inputs=latent_input,
        outputs=dec_output)

    # AUTOENCODEUR
    # Create autoencoder model
    autoencoder_model = keras.models.Model(
        inputs=input_layer,
        outputs=decoder_model(encoder_model(input_layer)))

    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    # Return encoder, decoder, autoencoder
    return encoder_model, decoder_model, autoencoder_model
