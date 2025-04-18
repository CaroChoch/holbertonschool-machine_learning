#!/usr/bin/env python3

"""
This module contains the Random_Forest class, which implements a random forest
of decision trees.
"""

import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """
    A class that implements a random forest of decision trees.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initialize the Random_Forest class.
        """
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predict the class of the explanatory data.
        """
        # Initialize an empty list to store predictions from individual trees
        all_predictions = []
        # Generate predictions for each tree in the forest
        for tree in self.numpy_preds:
            tree.update_predict()  # Make sure predict is initialized
            all_predictions.append(tree.predict(explanatory))
        predictions = np.array(all_predictions)

        # Calculate the mode (most frequent) prediction for each example
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(),
                                   axis=0,
                                   arr=predictions)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """
        Fit the Random_Forest class to the training data.
        """
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []
        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T)  # Stocke l'arbre entier
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}
    - Mean accuracy on training data : {np.array(accuracies).mean()}
    - Accuracy of the forest on td   : {self.accuracy(self.explanatory,
                                                      self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """
        Calculate the accuracy of the forest on the test data.
        """
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target))/test_target.size
