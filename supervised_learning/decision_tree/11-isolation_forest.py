#!/usr/bin/env python3
"""
This module contains the Isolation_Random_Forest class, which implements an
isolation random forest.
"""

import numpy as np
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest():
    """
    Isolation Random Forest class.
    """
    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """
        Initializes the isolation random forest.
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
        Predicts the anomaly score for each sample.
        """
        depths = np.array([f(explanatory) for f in self.numpy_preds])
        return depths.mean(axis=0)

    def fit(self, explanatory, n_trees=100, verbose=0):
        """
        Fits the isolation random forest to the data.
        """
        self.explanatory = explanatory
        self.target = None
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        np.random.seed(self.seed)  # Set global seed
        for i in range(self.n_trees):
            T = Isolation_Random_Tree(
                max_depth=self.max_depth,
                seed=self.seed+i)
            T.fit(explanatory)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.root.count_nodes_below())
            leaves.append(T.root.count_nodes_below(only_leaves=True))
        if verbose == 1:
            print(f"""  Training finished.
    - Mean depth                     : {np.array(depths).mean()}
    - Mean number of nodes           : {np.array(nodes).mean()}
    - Mean number of leaves          : {np.array(leaves).mean()}""")

    def suspects(self, explanatory, n_suspects):
        """
        Returns the n_suspects rows in explanatory that have
        the smallest depths (most anomalous)
        """
        depths = self.predict(explanatory)
        # Points with highest depths are considered anomalies
        suspect_indices = np.argsort(-depths)[:n_suspects]
        suspects = explanatory[suspect_indices]
        suspect_depths = depths[suspect_indices]
        # Format depths to 2 decimal places
        suspect_depths = np.round(suspect_depths, 2)
        # Format suspects coordinates to 8 decimal places
        suspects = np.round(suspects, 8)
        return suspects, suspect_depths
