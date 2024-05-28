#!/usr/bin/env python3
""" Set up the data pipeline """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Dataset class """
    def __init__(self, batch_size, max_len):
        """
        Class constructor
        Arguments:
            - batch_size is an integer representing the batch size for
                training/validation
            - max_len is an integer representing the maximum number of
                tokens allowed per example sentence
            """
        # Load the dataset with metadata and as supervised
        # `examples` contains the training and validation datasets
        # `metadata` contains information about the dataset such as
        # size, features, etc.
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        # Assign training and validation datasets
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.metadata = metadata  # Store metadata for later use
        self.batch_size = batch_size  # Store batch size
        self.max_len = max_len  # Store maximum sentence length

        # Create sub-word tokenizers from the training dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Tokenize the training and validation datasets
        # The tf_encode method converts each (pt, en) pair to tokenized format
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        # Setup data pipeline for training data
        # Filter out examples longer than `max_len` tokens
        self.data_train = self.data_train.filter(self.filter_max_len)
        # Cache the dataset to improve performance
        self.data_train = self.data_train.cache()
        # Shuffle the dataset to ensure random order
        self.data_train = self.data_train.shuffle(
            self.metadata.splits['train'].num_examples)
        # Split the dataset into padded batches of size `batch_size`
        self.data_train = self.data_train.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))
        # Prefetch the dataset to improve performance
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        # Setup data pipeline for validation data
        # Filter out examples longer than `max_len` tokens
        self.data_valid = self.data_valid.filter(self.filter_max_len)
        # Split the dataset into padded batches of size `batch_size`
        self.data_valid = self.data_valid.padded_batch(
            self.batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for dataset
        Arguments:
            - data is a tf.data.Dataset whose examples are formatted as a
              tuple (pt, en)
                - pt is the tf.Tensor containing the Portuguese sentence
                - en is the tf.Tensor containing the corresponding English
                  sentence
        Returns:
            tokenizer_pt, tokenizer_en
                - tokenizer_pt is the Portuguese tokenizer
                - tokenizer_en is the English tokenizer
        """
        # Create Portuguese tokenizer from the corpus
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus(
                (pt.numpy() for pt, en in data),
                target_vocab_size=2**15
            )

        # Create English tokenizer from the corpus
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.\
            build_from_corpus(
                (en.numpy() for pt, en in data),
                target_vocab_size=2**15
            )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens
        Arguments:
            - pt is the tf.Tensor containing the Portuguese sentence
            - en is the tf.Tensor containing the corresponding English sentence
        Returns:
            pt_tokens, en_tokens
                - pt_tokens is a tf.Tensor containing the Portuguese tokens
                - en_tokens is a tf.Tensor containing the English tokens
        """
        # Define start and end token indices for Portuguese
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        # Define start and end token indices for English
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1

        # Encode Portuguese sentence into tokens with start and end tokens
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        # Encode English sentence into tokens with start and end tokens
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the 'encode' instance method
        Arguments:
            - pt is the tf.Tensor containing the Portuguese sentence
            - en is the tf.Tensor containing the corresponding English sentence
        Returns:
            A tuple of the pt_tokens and en_tokens
        """
        # Use tf.py_function to wrap the 'encode' method
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        # Set the shape of the returned tensors
        pt_lang.set_shape([None])
        en_lang.set_shape([None])

        return pt_lang, en_lang

    def filter_max_len(self, pt, en):
        """
        Filters out all examples that have either sentence with more than
            max_len tokens
        Arguments:
            - pt: tf.Tensor containing the Portuguese tokens
            - en: tf.Tensor containing the English tokens
        Returns:
            - A boolean tensor indicating whether both sentences are
                within max_len
        """
        return tf.logical_and(tf.size(pt) <= self.max_len,
                              tf.size(en) <= self.max_len)
