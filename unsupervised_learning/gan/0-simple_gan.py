#!/usr/bin/env python3
""" Simple GAN """

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class SimpleGAN(keras.Model):
    """
    A simple Generative Adversarial Network (GAN) class that combines a
    generator and a discriminator.

    Attributes:
        - latent_generator (function): Function to generate latent space
            vectors.
        - real_examples (np.ndarray): Array of real examples for training.
        - generator (tf.keras.Model): The generator model.
        - discriminator (tf.keras.Model): The discriminator model.
        - batch_size (int): Size of the training batch.
        - disc_iter (int): Number of discriminator iterations per
            generator iteration.
        - learning_rate (float): Learning rate for the optimizers.
        - beta_1 (float): Beta 1 parameter for the Adam optimizer.
        - beta_2 (float): Beta 2 parameter for the Adam optimizer.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        """
        Initializes the SimpleGAN with the given parameters.
        """
        super().__init__()  # Run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .5  # Standard value, but can be changed if necessary
        self.beta_2 = .9  # Standard value, but can be changed if necessary

        # Define the generator loss and optimizer
        self.generator.loss = lambda x: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss)

        # Define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.

        Arguments:
            - size (int, optional): Size of the batch. Defaults to None.
            - training (bool, optional): Whether the generator is in training
                mode. Defaults to False.

        Returns:
            tf.Tensor: A batch of fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Selects a batch of real samples from the real_examples.

        Arguments:
            - size (int, optional): Size of the batch. Defaults to None.

        Returns:
            tf.Tensor: A batch of real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, data):
        """
        Performs one training step for the GAN.

        Arguments:
            - data: Not used, but required by the Keras API.

        Returns:
            dict: Dictionary containing the discriminator and generator loss.
        """
        # Training the discriminator
        for _ in range(self.disc_iter):
            real_samples = self.get_real_sample()
            fake_samples = self.get_fake_sample(training=True)

            with tf.GradientTape() as tape:
                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)

            gradients = tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(gradients, self.discriminator.trainable_variables))

        # Training the generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_output)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
