#!/usr/bin/env python3
""" Sparse Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates an sparse autoencoder
    Arguments:
        - input_dims: integer containing dimensions of model input
        - hidden_layers: list of integers containing number of nodes
            for each hidden layer in encoder, respectively
        - latent_dims: integer containing dimensions of latent space
        - lambtha: L1 regularization parameter
    Returns: encoder, decoder, auto
        - encoder: the encoder model
        - decoder: the decoder model
        - auto: the full autoencoder model
    """
    # ENCODER
    # Input layer
    input_layer = keras.Input(shape=(input_dims,))
    # Hidden layers
    enc_hidden = input_layer
    # Add hidden layers
    for layer_size in hidden_layers:
        enc_hidden = keras.layers.Dense(
            layer_size,
            activation='relu')(enc_hidden)
    # Latent representation with L1 regularization
    latent_representation = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha))(enc_hidden)
    # Create encoder model
    encoder_model = keras.models.Model(input_layer, latent_representation)

    # DECODER
    # Input layer
    latent_input = keras.Input(shape=(latent_dims,))
    # Hidden layers
    dec_hidden = latent_input
    # Add hidden layers (reversed order)
    for layer_size in reversed(hidden_layers):
        dec_hidden = keras.layers.Dense(
            layer_size,
            activation='relu')(dec_hidden)
    # Output layer
    output_layer = keras.layers.Dense(
        input_dims,
        activation='sigmoid')(dec_hidden)
    # Create decoder model
    decoder_model = keras.models.Model(
        latent_input,
        output_layer)

    # AUTOENCODEUR
    # Create autoencoder model
    autoencoder_model = keras.models.Model(
        inputs=input_layer,
        outputs=decoder_model(encoder_model(input_layer)))

    # Compile autoencoder
    autoencoder_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, autoencoder_model
