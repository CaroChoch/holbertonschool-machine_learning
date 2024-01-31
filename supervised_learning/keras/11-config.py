#!/usr/bin/env python3
""" Save and Load Configuration """
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format
    Arguments:
        - network: model whose configuration should be saved
        - filename: path of the file that the configuration should be saved to
    Returns: None
    """
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)
    return None


def load_config(filename):
    """
    loads a model with a specific configuration
    Arguments:
        - filename: path of the file containing the model’s configuration
          in JSON format
    Returns: the loaded model
    """
    with open(filename, 'r') as json_file:
        json_config = json_file.read()
    loaded_model = K.models.model_from_json(json_config)
    return loaded_model
