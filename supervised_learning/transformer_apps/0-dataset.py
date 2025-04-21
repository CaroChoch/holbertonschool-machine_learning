#!/usr/bin/env python3
""" Dataset class """
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
        Creates sub-word tokenizers for dataset using pretrained
        models adapted to our data
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in tfds.as_numpy(data):
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        pretrained_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True)
        pretrained_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True)

        tokenizer_pt = pretrained_pt.train_new_from_iterator(
            pt_sentences, vocab_size=2**13)
        tokenizer_en = pretrained_en.train_new_from_iterator(
            en_sentences, vocab_size=2**13)

        return tokenizer_pt, tokenizer_en
