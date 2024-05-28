#!/usr/bin/env python3
""" Encode a translation dataset into tokenized format """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Dataset class """
    def __init__(self):
        """ Class constructor """
        # Load the dataset with metadata and as supervised
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.metadata = metadata
        self.data_train = examples['train']  # training dataset
        self.data_valid = examples['validation']  # Validation dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)  # Create sub-word tokenizers

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
