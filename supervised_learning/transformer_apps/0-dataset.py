#!/usr/bin/env python3
""" Dataset class """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import transformers


class Dataset:
    """ Dataset class """

    def __init__(self):
        """ Class constructor """
        examples, metadata = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            with_info=True,
            as_supervised=True
        )
        self.metadata = metadata
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Charge les tokenizers pré-entraînés
        """
        tokenizer_pt = transformers.BertTokenizerFast.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.BertTokenizerFast.from_pretrained(
            'bert-base-uncased'
        )
        return tokenizer_pt, tokenizer_en
