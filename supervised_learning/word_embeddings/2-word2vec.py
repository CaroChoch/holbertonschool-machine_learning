#!/usr/bin/env python3
""" Creates and trains a gensim word2vec model """
import gensim


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0,
                   workers=1):
    """
    Creates and trains a gensim word2vec model
    Arguments:
        - sentences is a list of sentences to be trained on
        - size is the dimensionality of the embedding layer
        - min_count is the minimum number of occurrences of a word for
        use in training
        - window is the maximum distance between the current and predicted
        word within a sentence
        - negative is the size of negative sampling
        - cbow is a boolean to determine the training type; True is for CBOW;
        False is for Skip-gram
        - iterations is the number of iterations to train over
        - seed is the seed for the random number generator
        - workers is the number of worker threads to train the model
    Returns: the trained model
    """
    # Selecting training algorithm based on cbow parameter
    sg = 0 if cbow else 1  # 0 for CBOW, 1 for Skip-gram

    # Create and train the model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        iter=iterations,
        seed=seed,
        workers=workers,
    )

    model.save("word2vec.model")

    model = gensim.models.Word2Vec.load("word2vec.model")

    # Train the model
    model.train(sentences=sentences,
                total_examples=model.corpus_count,
                epochs=iterations)

    return model
