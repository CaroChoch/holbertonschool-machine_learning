#!/usr/bin/env python3
""" Wasserstein GANs """

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_clip(keras.Model):
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005):
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.clip_const = 1.0  # standard value but can be changed if necessary

        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: - tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=generator.optimizer,
            loss=generator.loss)

        # Define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: (
            -tf.math.reduce_mean(x) + tf.math.reduce_mean(y))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # Overloading train_step()
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

            # Clip the weights of the discriminator
            for weights in self.discriminator.trainable_variables:
                weights.assign(tf.clip_by_value(weights,
                                                -self.clip_const,
                                                self.clip_const))

        # Training the generator
        with tf.GradientTape() as tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_output)

        gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
