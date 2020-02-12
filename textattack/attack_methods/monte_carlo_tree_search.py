from .attack import Attack
from textattack.attack_results import AttackResult, FailedAttackResult
import numpy as np
import math, random, statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

NODE_ID = 0

DEBUG = True
import cProfile, pstats, io
from pstats import SortKey

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
        self.local_rave_values = {} # Maps action (int, str) --> value (int)
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
        self.actions_to_skip = set()

        self.game_value = 0.0
        self.global_rave_values = {} # Maps action (int, str) --> value (int)

        self.terminate = False  # probably won't need this

    def reset_tree(self):
        self.available_words_to_replace = set(range(len(original_text.words)))
        self.game_value = 0.0
        self.terminate = False

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


    def generate_transformations(self, tokenized_text, indices_to_replace):
        # Uses `get_transformations` and add item to TransformationCache
        transformed_texts = self.get_transformations(
            tokenized_text,
            original_text=self.tree.original_text,
            indices_to_replace=indices_to_replace
        )
        transformed_words = [text.attack_attrs['new_word'] for text in transformed_texts]

        self


    def choose_random_action(self):
        if self.tree.available_actions:
            return random.choice(tuple(self.tree.available_actions))
        else:
            return None

    def evaluate_transformation(self, transformed_text):
        prob_scores = self._call_model([transformed_text])[0]
        new_label = prob_scores.argmax().item()
        value = self.tree.orignal_score - prob_scores[self.tree.original_label]

        return new_label, value
    
    def evaluate_action(self, current_text, action):
        transformations = self.get_transformations(current_text, indices_to_replace=[action])
        if len(transformations) != 0:
            prob_scores = self._call_model(transformations)
            best_index = prob_scores[:, self.tree.original_label].argmin()
            orig_label_removed_scores = torch.cat((prob_scores[best_index][:self.tree.original_label], 
                                            prob_scores[best_index][self.tree.original_label+1:])
                                        )

            value = orig_label_removed_scores.max().item() - prob_scores[best_index][self.tree.original_label].item()
            new_label = prob_scores[best_index].argmax().item()

            return transformations[best_index], new_label, value
        else:
            return current_text, self.tree.original_label, -1

    def backprop(self, current_node):
        while current_node is not None:
            current_node.value += self.tree.game_value
            current_node.num_visits += 1
            current_node = current_node.parent

    def simulate(self, current_node, current_text):
        current_depth = current_node.depth

        while not self.tree.terminate:
            action = self.choose_random_action()
            current_text, new_label, value = self.evaluate_action(current_text, action)
            self.tree.available_actions.remove(action)

            if new_label != self.tree.original_label:
                self.tree.terminate = True
                self.game_value = value
            elif current_depth == self.tree.max_depth:
                self.tree.terminate = True
                self.game_value = value

    def expansion(self, current_node):
        action = self.choose_random_action()
        current_node.children[action] = Node(current_node.depth+1, current_node)
        current_node = current_node.children[action]
        self.tree.available_actions.remove(action)

        current_text, new_label, value = self.evaluate_action(current_text, action)

        if new_label != self.tree.original_label:
            self.tree.terminate = True
            self.game_value = value

        else:
            self.tree.terminate = True
            _, value = self.evaluate_transformation(current_text)
            self.game_value = value

        return current_node, current_text

    def UCB(self, node, parent_num_visits):
        return node.value + math.sqrt(self.ucb_C * math.log(parent_num_visits) / node.num_visits)

    def UCB_RAVE_tuned(self, node, action, parent_num_visits):
        ucb = self.ucb_C * math.log(parent_num_visits) / node.num_visits
        return node.value + self.tree.global_rave_values[action] + self.node.local_rave_values[action]
                + math.sqrt(ucb * min(0.25, node.variance + math.sqrt(ucb)))
    
    def selection(self):

        current_node = self.tree.root
        current_text = self.tree.root.state

        best_next_node = None
        best_ucb_value = float('-inf')
        best_action = None

        while not current_node.is_leaf():

            for action in current_node.children.keys():
                ucb_value = self.UCB_RAVE_tuned(current_node.children[action], action, current_node.num_visits)
 
                if ucb_value > best_ucb_value:
                    best_next_node = current_node.children[action]
                    best_ucb_value = ucb_value
                    best_action = action

            current_node = best_next_node
            self.tree.available_actions.remove(best_action)

        return current_node

    def run_mcts(self):

        for i in range(self.num_iter):
            self.tree.reset_tree()
            current_node = self.selection()
            
            if current_node.depth < 5:
                current_node = self.expansion(current_node)

            if not self.tree.terminate:
                self.simulate(current_node, current_text)

            self.backprop(current_node)

        # Select best choice based on node value
        best_action = None
        best_value = float('-inf')
        for action in self.tree.root.children.keys():
            node_value = self.tree.root.children[action].value 
            if node_value > best_value:
                best_action = action
                best_value = node_value

        return best_action

    def attack_one(self, original_label, tokenized_text):

        original_tokenized_text = tokenized_text
        original_confidence = self._call_model([original_tokenized_text]).squeeze()[original_label]
        max_tree_depth = min(self.max_words_changed, len(tokenized_text.words))

        self.tree = SearchTree(tokenized_text, original_label, original_confidence, max_tree_depth)

        replacements = self.transformation._get_replacement_words(tokenized_text.words[2])

        print(replacements)

        profiler = cProfile.Profile()
        profiler.enable()
        replacements = self.get_transformations(tokenized_text, indices_to_replace=[0,1,2,3,4])
        profiler.disable()

        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())

        """
        for i in range(max_words_changed):
            best_next_action = self.run_mcts(
                                original_label, 
                                original_confidence_score, 
                                tokenized_text,
                                max_words_changed
                            )

            tokenized_text, new_label, _ = self.evaluate_action(tokenized_text, best_next_action)

            if new_label != original_label:
                break
        """

        if original_label == new_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult( 
                original_tokenized_text, 
                tokenized_text, 
                original_label,
                new_label
            )