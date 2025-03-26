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
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def update_bounds_below(self):
        """ Update the bounds of the node """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child:
                child.upper = self.upper.copy()
                child.lower = self.lower.copy()

                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                elif child == self.right_child:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """
        Updates the node's indicator that determines if a point belongs to
        the node's region.
        The indicator checks if a point x:
        1. Is greater than all lower bounds
        2. Is less than or equal to all upper bounds
        """
        def is_large_enough(x):
            """Checks if points are above the lower bounds"""
            # For each feature (key) in lower, check if x is greater
            # than the bound
            conditions = []
            for feature in self.lower:
                condition = np.greater(x[:, feature], self.lower[feature])
                conditions.append(condition)
            # Returns True if all conditions are met
            return np.all(np.array(conditions), axis=0)

        def is_small_enough(x):
            """Checks if points are below the upper bounds"""
            # For each feature (key) in upper,check if x is less than the bound
            conditions = []
            for feature in self.upper:
                condition = np.less_equal(x[:, feature], self.upper[feature])
                conditions.append(condition)
            # Returns True if all conditions are met
            return np.all(np.array(conditions), axis=0)

        # The final indicator combines both conditions
        self.indicator = lambda x: np.all(np.array([
            is_large_enough(x),  # Points above lower bounds
            is_small_enough(x)   # Points below upper bounds
        ]), axis=0)

    def pred(self, x):
        """ Predict the value of the node """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """ Get all leaves below the leaf """
        return [self]

    def update_bounds_below(self):
        """ Update the bounds of the leaf """
        pass

    def pred(self, x):
        """ Predict the value of the leaf """
        return self.value


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

    def update_bounds(self):
        """ Update the bounds of the decision tree """
        self.root.update_bounds_below()

    def update_predict(self):
        """ Update the prediction of the decision tree """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_single(x):
            """Helper function to predict a single point"""
            for leaf in leaves:
                if leaf.indicator(x.reshape(1, -1)):
                    return leaf.value
            return None  # or some default value if no leaf matches

        self.predict = lambda A: np.array([predict_single(x) for x in A])

    def pred(self, x):
        """ Predict the value of the decision tree """
        return self.root.pred(x)

    def fit(self,explanatory, target,verbose=0):
        """ Fit the decision tree to the data """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else: 
            self.split_criterion = self.Gini_split_criterion  ## <--- to be defined later
        self.explanatory = explanatory
        self.target      = target
        self.root.sub_population = np.ones_like(self.target,dtype='bool')

        self.fit_node(self.root)  ##   <--- to be defined later

        self.update_predict() ##    <--- defined in the previous task

        if verbose==1:
            print(f"""  Training finished.
                - Depth                     : { self.depth()       }
                - Number of nodes           : { self.count_nodes() }
                - Number of leaves          : { self.count_nodes(only_leaves=True) }
                - Accuracy on training data : { self.accuracy(self.explanatory,self.target)    }""")  ##   <--- to be defined later

    def np_extrema(self,arr):
        return np.min(arr), np.max(arr)

    def random_split_criterion(self,node):
        """ Random split criterion """
        diff=0
        while diff==0 :
            feature=self.rng.integers(0,self.explanatory.shape[1])
            feature_min,feature_max=self.np_extrema(self.explanatory[:,feature][node.sub_population])
            diff=feature_max-feature_min
        x=self.rng.uniform()
        threshold= (1-x)*feature_min + x*feature_max
        return feature,threshold

    def fit_node(self,node):
        """ Fit the node """
        node.feature, node.threshold = self.split_criterion(node)

        left_population = np.logical_and(node.sub_population, 
            self.explanatory[:, node.feature] <= node.threshold)
        right_population = np.logical_and(node.sub_population,
            self.explanatory[:, node.feature] > node.threshold)

        # Is left node a leaf ?
        is_left_leaf = np.sum(left_population) <= self.min_pop

        if is_left_leaf:
                node.left_child = self.get_leaf_child(node,left_population)                                                         
        else:
                node.left_child = self.get_node_child(node,left_population)
                self.fit_node(node.left_child)

        # Is right node a leaf ?
        is_right_leaf = np.sum(right_population) <= self.min_pop

        if is_right_leaf:
                node.right_child = self.get_leaf_child(node,right_population)
        else:
                node.right_child = self.get_node_child(node,right_population)
                self.fit_node(node.right_child)    

    def get_leaf_child(self, node, sub_population):
        """ Get the leaf child """
        value = np.mean(self.target[sub_population])
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """ Get the node child """
        n= Node()
        n.depth=node.depth+1
        n.sub_population=sub_population
        return n

    def accuracy(self, test_explanatory , test_target):
        """ Calculate the accuracy of the decision tree """
        return np.sum(np.equal(self.predict(test_explanatory), test_target))/test_target.size
