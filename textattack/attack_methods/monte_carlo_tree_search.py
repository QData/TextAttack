from .attack import Attack
from textattack.attack_results import AttackResult, FailedAttackResult
import numpy as np
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

DEBUG = True

class Node:

    def __init__(self, depth, parent):
        self.depth = depth
        self.parent = parent
        self.num_visits = 0
        self.value = 0.0
        self.children = {}

    def is_leaf(self):
        return not bool(self.children)

class SearchTree:

    def __init__(self, original_text, original_label, original_score, max_words_changed):
        self.root = Node(0, None)
        self.original_text = original_text
        self.original_label = original_label
        self.orignal_score = original_score
        self.max_depth = min(max_words_changed, len(original_text.words))

        self.unchanged_words = set(range(len(original_text.words)))
        self.game_value = 0.0
        self.terminate = False

    def reset_tree(self):
        self.unchanged_words = set(range(len(original_text.words)))
        self.game_value = 0.0
        self.terminate = False

class MonteCarloTreeSearch(Attack):
    """ 
    Uses Monte Carlo Tree Search (MCTS) to attempt to find the most important words in an input.
    Args:
        model: A PyTorch or TensorFlow model to attack.
        transformation: The type of transformation to use. Should be a subclass of WordSwap. 
        constraints: A list of constraints to add to the attack
        reward_type (str): Defines what type of function to use for MCTS.
            - prob_diff: Uses "max_{c'} P(c'|s) - P(c_l|s)"
            - entropy: Uses negative log-likelihood function
        max_iter (int) : Maximum iterations for MCTS. Default is 4000
        max_words_changed (int) : Maximum number of words we change during MCTS. Effectively represents depth of search tree.
    """
    def __init__(self, model, transformation, constraints=[], 
        reward_type="prob_diff", max_iter=50, max_words_changed=5
    ):
        super().__init__(model, transformation, constraints=constraints)
        self.reward_type = reward_type
        self.max_iter = max_iter
        self.max_words_changed = max_words_changed
        self.bestfeature = []
        self.ucb_C = 2


    def choose_random_action(self):
        return random.choice(tuple(self.tree.unchanged_words))


    # def play_action(self, current_text, action):
    #     """
    #     Generate transformation for the chosen word of current text and choose the best one.

    #     Args:
    #         action: index of the word to transform

    #     Returns: TokenizedText object, list of scores
    #     """

    #     transformations = self.get_transformations(current_text, indices_to_replace=[action])
    #     if len(transformations) != 0:
    #         scores = self._call_model(transformations)
    #         best_index = scores[:, self.tree.original_label].argmin()
    #         return transformations[best_index], scores[best_index]
    #     else:
    #         return current_text, None

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

            if new_label != self.tree.original_label:
                self.tree.terminate = True
                self.game_value = value
            elif current_depth == self.tree.max_depth:
                self.tree.terminate = True
                self.game_value = value

    def expansion(self, current_node, current_text):
        if current_node.depth < self.tree.max_depth:
            action = self.choose_random_action()
            current_node.children[action] = Node(current_node.depth+1, current_node)
            current_node = current_node.children[action]
            self.tree.unchanged_words.remove(action)

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

    def selection(self):

        current_node = self.tree.root
        current_text = self.tree.original_text
        best_next_node = None
        best_ucb_value = float('-inf')
        best_action = None

        while not (current_node.is_leaf() or self.tree.terminate):

            for action in current_node.children.keys():
                ucb_value = self.UCB(current_node.children[action], current_node.num_visits)

                if ucb_value > best_ucb_value:
                    best_next_node = current_node.children[action]
                    best_ucb_value = ucb_value
                    best_action = action

            current_node = best_next_node

            # Visiting the current node
            current_text, new_label, value = self.evaluate_action(current_text, best_action)
            self.tree.unchanged_words.remove(best_action)

            if new_label != self.tree.original_label:
                self.tree.terminate = True
                self.tree.game_value = value

        return current_node, current_text

    def run_mcts(self, original_label, original_score, tokenized_text):

        self.tree = SearchTree(tokenized_text, original_label, original_score, self.max_words_changed)

        for i in range(self.max_iter):
            self.tree.reset_tree()
            current_node, current_text = self.selection()
            
            if not self.tree.terminate:
                current_node, current_text = self.expansion(current_node, current_text)

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
        original_prob = self._call_model([original_tokenized_text]).squeeze()
        original_score = original_prob[original_label]
        max_words_changed = min(self.max_words_changed, len(tokenized_text.words))

        for i in range(max_words_changed):
            best_next_action = self.run_mcts(original_label, original_score, tokenized_text)

            tokenized_text, new_label, _ = self.evaluate_action(tokenized_text, best_next_action)

            if new_label != original_label:
                break

        if original_label == new_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult( 
                original_tokenized_text, 
                tokenized_text, 
                original_label,
                new_label
            )