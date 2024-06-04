#!/usr/bin/env python3
""" Wasserstein GANs with gradient penalty """

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class WGAN_GP(keras.Model):
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
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the Wasserstein GANs with gradient penalty with
        the given parameters.
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

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

    # generator of fake samples of size batch_size
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

    # generator of interpolating samples of size batch_size
    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates a batch of interpolated samples between real and fake samples
        Arguments:
            - real_sample (tf.Tensor): A batch of real samples.
            - fake_sample (tf.Tensor): A batch of fake samples.
        Returns:
            tf.Tensor: A batch of interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    # computing the gradient penalty
    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the interpolated samples.
        Arguments:
            - interpolated_sample (tf.Tensor): The interpolated samples.
        Returns:
            tf.Tensor: The gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

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
                
                

                # get the interpolated sample
                interpolated_sample = self.get_interpolated_sample(
                    real_samples, fake_samples)
                

                real_output = self.discriminator(real_samples, training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_output, fake_output)

                # compute the gradient penalty
                gp = self.gradient_penalty(interpolated_sample)
                discr_loss += self.lambda_gp * gp

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
