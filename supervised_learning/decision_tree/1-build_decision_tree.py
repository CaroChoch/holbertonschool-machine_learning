#!/usr/bin/env python3
""" Number of nodes/leaves in a decision tree """

import numpy as np


class Node:
    """ Class that represents a node in a decision tree """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """ Class constructor """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """ Calculate the depth of the node """
        if self.is_leaf:
            return self.depth

        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below the node """
        left_count = self.left_child.count_nodes_below(only_leaves=only_leaves)
        right_count = self.right_child.count_nodes_below(
            only_leaves=only_leaves)
        if only_leaves:
            return left_count + right_count
        return 1 + left_count + right_count


class Leaf(Node):
    """ Class that represents a leaf in a decision tree """
    def __init__(self, value, depth=None):
        """ Class constructor """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """ Calculate the depth of the leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """ Count the number of nodes below the leaf """
        return 1


class Decision_Tree():
    """ Class that represents a decision tree """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """ Class constructor """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """ Calculate the depth of the decision tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """ Count the number of nodes in the decision tree """
        return self.root.count_nodes_below(only_leaves=only_leaves)
