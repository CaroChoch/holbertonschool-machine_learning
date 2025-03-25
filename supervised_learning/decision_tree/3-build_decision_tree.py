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

    def __str__(self):
        """
        Prints string representation of the node and its children.
        """
        if self.is_root:
            s = "root"
        else:
            s = "node"
        s = f"{s} [feature={self.feature},"
        s += f" threshold={self.threshold}]\n"

        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            s += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            s += f"\n    +---> {right_str}"

        return s.rstrip()

    def left_child_add_prefix(self, text):
        """
        Adds the string representation of the left child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """
        Adds the string representation of the right child to the given text
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return new_text

    def get_leaves_below(self):
        """ Returns the list of all leaves of the tree """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves


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

    def __str__(self):
        """
        Returns a string representation of the leaf.
        """
        return (f"leaf [value={self.value}]")

    def get_leaves_below(self):
        """ Get all leaves below the leaf """
        return [self]


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

    def __str__(self):
        """
        Returns a string representation of the entire decision tree.
        """
        # rstrip() to remove the extra newline at the end of the string
        return self.root.__str__() + "\n"

    def get_leaves(self):
        """ Get all leaves in the decision tree """
        return self.root.get_leaves_below()
