#!/usr/bin/env python3
"""Dataset class with TensorFlow encoding wrapper."""
import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """Dataset class for Portuguese-English translations."""

    def __init__(self):
        """Class constructor: load data, build tokenizers, and encode examples."""
        # Load the TED talk translation dataset with info
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        self.metadata = metadata
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Build subword tokenizers from a subset to speed up
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        # Apply the TensorFlow wrapper for encoding
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for dataset using pretrained
        models adapted to our data via training-from-iterator.
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Initialize pretrained tokenizers and train on our corpus
        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True
        )
        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True
        )

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13
        )
        tokenizer_en = pretrained_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encode a pair of sentences into token IDs, including start and end tokens.
        """
        # Start and end token IDs
        pt_start = self.tokenizer_pt.vocab_size
        pt_end = pt_start + 1
        en_start = self.tokenizer_en.vocab_size
        en_end = en_start + 1

        # Convert bytes to string
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        # Tokenize and add start/end
        pt_tokens = [pt_start] + self.tokenizer_pt.encode(pt_text) + [pt_end]
        en_tokens = [en_start] + self.tokenizer_en.encode(en_text) + [en_end]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper around the `encode` method.
        Returns two tf.int64 tensors with shape [None].
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        # Set static shape for TensorFlow
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])
        return pt_tokens, en_tokens
