import io
import pstats
import cProfile
from pstats import SortKey
from .attack import Attack
from textattack.attack_results import AttackResult, FailedAttackResult
import numpy as np
import math
import random
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

NODE_ID = 0

DEBUG = True


class TransformationCache:

    def __init__(self):
        self.cache = {}

    def get_transformation(i, new_word):
        """
        Args:
            i (int) : i-th word to transform
            word (str): transformed word
        Returns:
            TokenizedText
        """
        if i in self.cache.keys() and word in self.cache[i].keys() > 0:
            return self.cache[i][new_word]
        else:
            return None

    def get_random_transformation(indices_to_transform):
        i = random.choice(indices_to_transform)
        if i in self.cache.keys():
            words

        if word in self.cache.keys() and transformation in self.cache[i]
        if word in self.cache.keys() and len(self.cache[word]) > 0:
            return random.choice(self.cache[word])
        else:
            return None

    def store_transformation(word, transformations):
        self.cache[word] = transformations


class Node:

    """
        Represents a node in search tree
        Members:
            state (TokenizedText) : Current version of TokenizedText
            action_seq : list of actions that led to current state
            depth : Current depth in search tree
            parent : Parent node
            num_visits : Number of visits to the current node
            ...
    """

    def __init__(self, text, depth, parent):
        self.id = NODE_ID
        NODE_ID += 1
        self.state = text
        self.depth = depth
        self.parent = parent
        self.num_visits = 0
        self.value = 0.0
        self.local_rave_values = {}  # Maps action (int, str) --> value (int)
        self.value_history = []
        self.variance = 0.0
        self.children = {}

    def is_leaf(self):
        return not bool(self.children)

    def calc_variance(self):
        self.variance = statistics.variance(self.value_history, self.value)


class SearchTree:
    """
        Object used to hold states/variables for a specific run of MCTS
    """

    def __init__(self, original_text, original_label, original_confidence, max_depth):
        self.root = Node(original_text, 0, None)
        self.original_text = original_text
        self.original_label = original_label
        self.orignal_confidence = original_confidence
        self.max_depth = max_depth

        self.available_words_to_replace = set(range(len(original_text.words)))
        self.words_to_skip = set()
        self.action_history = []
        self.iteration = 0

        self.global_rave_values = {}  # Maps action (int, str) --> value (int)

    def reset_tree(self):
        self.available_words_to_replace = set(range(len(original_text.words)))
        self.action_history = []


class MonteCarloTreeSearch(Attack):
    """ 
    Uses Monte Carlo Tree Search (MCTS) to attempt to find the most important words in an input.
    Args:
        model: A PyTorch or TensorFlow model to attack.
        transformation: The type of transformation to use. Should be a subclass of WordSwap. 
        constraints: A list of constraints to add to the attack
            - entropy: Uses negative log-likelihood function
        num_iter (int) : Number of iterations for MCTS. Default is 4000
        max_words_changed (int) : Maximum number of words we change during MCTS. Effectively represents depth of search tree.
    """

    def __init__(self, model, transformation, constraints=[], num_iter=1000, max_words_changed=5):
        super().__init__(model, transformation, constraints=constraints)

        # MCTS Hyper-parameters
        self.num_iter = num_iter
        self.max_words_changed = max_words_changed
        self.ucb_C = 2

        self.word_embedding_distance = 0

    def evaluate(current_node):
        """
            Evaluates the final (or current) transformation
        """
        prob_scores = self._call_model([current_node.text])[0]
        new_label = prob_scores.argmax().item()

        value = (new_label != self.tree.original_label) + \
            self.tree.original_confidence - \
            prob_scores[self.tree.original_label]

        constraint_scores = current_node.text.attack_attrs['constraint_scores']
        for key in constraints_scores.keys():
            value += constraints_scores[key]

        return value, new_label

    def backprop(self, current_node, search_value):
        """
            Update score statistics for each node, starting from last leaf node of search iteration
                and ending at the root.
            Also update global RAVE values. No return value.
        """

        # Update global RAVE values
        for action in self.tree.action_history:
            self.tree.global_rave_values[action] = (
                self.tree.global_rave_values[action] + search_value) / self.tree.iteration

        while current_node is not None:
            current_node.num_visits += 1
            current_node.value = (current_node.value +
                                  search_value) / current_node.num_visits
            current_node.value_history.append(search_value)
            current_node.calc_variance()

            # Update local RAVE values
            for action in self.tree.action_history:
                current_node.local_rave_values[action] = (
                    current_node.local_rave_values[action] + search_value) / current_node.num_visits

            current_node = current_node.parent

    def simulate(self, current_node):

        while current_node.depth < self.tree.max_depth:
            if self.tree.available_transformations:
                break

            random_tranformation = None
            while random_tranformation is None and self.tree.available_transformations:
                random_word_to_replace = random.choice(
                    tuple(self.tree.available_words_to_replace))

                self.tree.available_words_to_replace.remove(
                    random_word_to_replace)

                available_transformations = self.get_transformations(
                    current_node.text,
                    original_text=self.tree.original_text,
                    indices_to_replace=[random_word_to_replace]
                )

                if not available_transformations:
                    continue

                random_tranformation = random.choice(available_transformations)

            if random_tranformation is None:
                break

            self.tree.action_history.append(
                (random_word_to_replace, random_tranformation.attack_attrs['new_word']))
            current_node = Node(random_tranformation,
                                current_node+1, current_node)

        return current_node

    def expansion(self, current_node):
        """
            Create next nodes based on available transformations and then take a random action.
            Returns: New node that we expand to. If no such node exists, return None
        """
        words_to_replace = list(
            self.tree.available_words_to_replace.difference(self.tree.words_to_skip))
        available_transformations = self.get_transformations(
            current_node.text,
            original_text=self.tree.original_text,
            indices_to_replace=words_to_replace
        )

        if len(available_transformations) == 0:
            self.tree.words_to_skip = self.tree.words_to_skip.union(
                set(words_to_replace))
            return None
        else:

            available_actions = [(t.attack_attrs['modified_word_index'],
                                  t.attack_attrs['new_word']) for t in available_transformations]

            for i in range(len(available_actions)):
                # Create children nodes if it doesn't exist already
                current_node.children[available_actions[i]] = Node(
                    available_transformations[i], current_node.depth+1, current_node)

            random_action = random.choice(available_actions)

            self.tree.available_words_to_replace.remove(random_action[0])
            self.tree.action_history.append(random_action)

            return current_node.children[random_action]

    def UCB(self, node, parent_num_visits):
        return node.value + math.sqrt(self.ucb_C * math.log(parent_num_visits) / node.num_visits)

    def UCB_RAVE_tuned(self, node, action, parent_num_visits):
        ucb = self.ucb_C * math.log(parent_num_visits) / node.num_visits
        return node.value + self.tree.global_rave_values[action] + self.node.local_rave_values[action]
        + math.sqrt(ucb * min(0.25, node.variance + math.sqrt(ucb)))

    def selection(self):
        """
            Select the best next node according to UCB function. Finish when node is a leaf
            Returns last node of selection process.
        """

        current_node = self.tree.root
        current_text = self.tree.root.state

        best_next_node = None
        best_ucb_value = float('-inf')
        best_action = None

        while not current_node.is_leaf():

            for action in current_node.children.keys():
                ucb_value = self.UCB_RAVE_tuned(
                    current_node.children[action], action, current_node.num_visits)

                if ucb_value > best_ucb_value:
                    best_next_node = current_node.children[action]
                    best_ucb_value = ucb_value
                    best_action = action

            current_node = best_next_node
            self.tree.available_words_to_replace.remove(best_action[0])
            self.tree.action_history.append(best_action)

        return current_node

    def run_mcts(self):
        """
            Runs Monte Carlo Tree Search at the current root.
            Returns best action
        """

        for i in range(self.num_iter):
            self.tree.iteration += 1
            self.tree.reset_tree()
            current_node = self.selection()

            if current_node.depth < self.tree.max_depth:
                current_node = self.expansion(current_node)

            if current_node is not None and current_node.depth < self.tree.max_depth:
                current_node = self.simulate(current_node)

            search_value, _ = self.evaluate(current_node)
            self.backprop(current_node, search_value)

        # Select best choice based on node value
        best_action = max(self.tree.root.children,
                          key=self.tree.root.children.get.value)

        return self.tree.root.children[best_action]

    def attack_one(self, original_label, tokenized_text):

        original_tokenized_text = tokenized_text
        original_confidence = self._call_model(
            [original_tokenized_text]).squeeze()[original_label]
        max_tree_depth = min(self.max_words_changed, len(tokenized_text.words))

        self.tree = SearchTree(
            tokenized_text, original_label, original_confidence, max_tree_depth)

        """
        profiler = cProfile.Profile()
        profiler.enable()
        replacements = self.get_transformations(
            tokenized_text, indices_to_replace=[0, 1, 2, 3, 4])
        profiler.disable()

        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        """
        for i in range(max_words_changed):
            best_next_node = self.run_mcts()

            self.tree.root = best_next_node

            _, new_label = self.evaluate(self.tree.root)

            if new_label != original_label:
                new_tokenized_text = self.tree.root.text
                break

        if original_label == new_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult(
                original_tokenized_text,
                new_tokenized_text,
                original_label,
                new_label
            )
