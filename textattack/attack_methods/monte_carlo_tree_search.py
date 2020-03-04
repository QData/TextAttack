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
import time
import collections
import multiprocessing as mp

PROFILE = False
if PROFILE:
    import io
    import pstats
    import cProfile
    from pstats import SortKey


LOG_OUPUTS = True
if LOG_OUPUTS:
    import graphviz
    import pickle

def get_edge_label(root, action):
    root_text = root.text.words
    return f"({root_text[action[0]]} --> {action[1]})"

def get_node_label(node):
    value = str(round(node.value,2))
    return value + " / " + str(node.num_visits)

def generate_dot(root_node, action_history, filename, success):

    dot = graphviz.Graph(f"mcts_{time.time()}", comment=f"{root_node.text.text}", filename=f"mcts_{time.time()}")
    queue = collections.deque()
    queue.append(root_node)

    dot.attr(label=f"\n\n\nOriginal text: {root_node.text.text}")
    dot.node(str(root_node.id), label=get_node_label(root_node))

    realized_node = root_node.id

    if success:
        color = "green"
    else:
        color = "red"

    while queue:
        node = queue.popleft()

        for action, child in node.children.items():
            if child.num_visits > 0:
                if action_history and action == action_history[0] and node.id == realized_node:
                    dot.node(str(child.id), label=get_node_label(child), color=color)
                    dot.edge(str(node.id), str(child.id), label=get_edge_label(root_node, action), color=color)
                    action_history.pop(0)
                    realized_node = child.id
                else:
                    dot.node(str(child.id), label=get_node_label(child))
                    dot.edge(str(node.id), str(child.id), label=get_edge_label(root_node, action))
                queue.append(child)

    with open(f"outputs/{filename}.pkl", "wb") as f:
        pickle.dump(dot, f)


NODE_ID = 1

class Node:

    """
        Represents a node in search tree
        Members:
            text (TokenizedText) : Current version of TokenizedText
            action_seq : list of actions that led to current state
            depth : Current depth in search tree
            parent : Parent node
            num_visits : Number of visits to the current node
            ...
    """

    def __init__(self, node_id, text, depth, parent):
        self.id = node_id
        self.text = text
        self.depth = depth
        self.parent = parent
        self.num_visits = 0
        self.value = 0.0
        # Maps action (int, str) --> value (int)
        self.local_rave_values = {}
        self.value_history = []
        self.variance = 0.0
        self.children = {}

    def is_leaf(self):
        return not bool(self.children)

    def calc_variance(self):
        if len(self.value_history) >= 2:
            self.variance = statistics.variance(self.value_history, self.value)


class SearchTree:
    """
        Object used to hold states/variables for a specific run of MCTS
    """

    def __init__(self, original_text, original_label, original_confidence, max_depth):
        self.root = Node(0, original_text, 0, None)
        self.original_text = original_text
        self.original_label = original_label
        self.original_confidence = original_confidence
        self.max_depth = max_depth

        self.available_words_to_replace = set(
            range(len(self.original_text.words)))
        self.words_to_skip = set()
        self.action_history = []
        self.iteration = 0

        # Maps action (int, str) --> (value (int), number_values (int))
        self.global_rave_values = {}

    def reset_tree(self):
        self.available_words_to_replace = set(
            range(len(self.original_text.words)))
        self.action_history = []

    def reset_node_depth(self):
        queue = collections.deque()
        self.root.depth = 0
        queue.append(self.root)

        while queue:
            node = queue.popleft()

            for action, child in node.children.items():
                child.dept = node.depth + 1
                queue.append(child)

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

    def __init__(self, model, transformation, constraints=[], num_iter=500, max_words_changed=10):
        super().__init__(model, transformation, constraints=constraints)

        # MCTS Hyper-parameters
        self.num_iter = num_iter
        self.max_words_changed = max_words_changed
        self.ucb_C = 5
        self.global_C = 50
        self.local_C = 50
        self.window_size = 6

        self.word_embedding_distance = 0

    def evaluate(self, current_node):
        """
            Evaluates the final (or current) transformation
        """
        prob_scores = self._call_model([current_node.text])[0]
        new_label = prob_scores.argmax().item()

        value = self.tree.original_confidence - prob_scores[self.tree.original_label].item()

        value += sum(
            current_node.text.attack_attrs['constraint_scores'].values())

        return value, new_label

    def backprop(self, current_node, search_value):
        """
            Update score statistics for each node, starting from last leaf node of search iteration
                and ending at the root.
            Also update global RAVE values. No return value.
        """

        # Update global RAVE values
        for action in self.tree.action_history:
            if action in self.tree.global_rave_values:
                old_rave = self.tree.global_rave_values[action]
                new_value = (old_rave[0] * old_rave[1] + search_value) / (old_rave[1] + 1)

                self.tree.global_rave_values[action] = (new_value, old_rave[1] + 1)
            else:
                self.tree.global_rave_values[action] = (search_value, 1)
                

        while current_node is not None:
            n = current_node.num_visits
            current_node.num_visits += 1
            current_node.value = (current_node.value * n + search_value) / (n + 1)
            current_node.value_history.append(search_value)
            current_node.calc_variance()

            # Update local RAVE values
            for action in self.tree.action_history:
                if action in current_node.local_rave_values:
                    current_node.local_rave_values[action] = (current_node.local_rave_values[action] \
                        * n + search_value) / (n + 1)
                else:
                    current_node.local_rave_values[action] = search_value

            current_node = current_node.parent

    def simulate(self, current_node):

        while current_node.depth < self.tree.max_depth:
            if self.tree.available_words_to_replace:
                break

            random_tranformation = None
            while random_tranformation is None and self.tree.available_words_to_replace:
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
            current_node = Node(-1, random_tranformation,
                                current_node+1, current_node)

        return current_node

    def expansion(self, current_node):
        """
            Create next nodes based on available transformations and then take a random action.
            Returns: New node that we expand to. If no such node exists, return None
        """
        global NODE_ID
        words_to_replace = list(self.tree.available_words_to_replace)

        available_transformations = self.get_transformations(
            current_node.text,
            original_text=self.tree.original_text,
            indices_to_replace=words_to_replace
        )

        if len(available_transformations) == 0:
            return current_node
        else:

            available_actions = [(t.attack_attrs['modified_word_index'],
                                  t.attack_attrs['new_word']) for t in available_transformations]

            for i in range(len(available_actions)):
                # Create children nodes if it doesn't exist already
                current_node.children[available_actions[i]] = Node(
                    NODE_ID, available_transformations[i], current_node.depth+1, current_node)
                NODE_ID += 1

            random_action = random.choice(available_actions)

            self.tree.available_words_to_replace.remove(random_action[0])
            self.tree.action_history.append(random_action)

            return current_node.children[random_action]

    def UCB(self, node, parent_num_visits):
        return node.value + math.sqrt(self.ucb_C * math.log(parent_num_visits) / node.num_visits)

    def UCB_RAVE_tuned(self, node, action, parent_num_visits):
        ucb = self.ucb_C * math.log(parent_num_visits) / max(1, node.num_visits)
        value = node.value + \
            math.sqrt(ucb * min(0.25, node.variance + math.sqrt(ucb)))

        if action in self.tree.global_rave_values:
            alpha = self.global_C / (self.global_C + self.tree.global_rave_values[action][1])
            value += alpha * self.tree.global_rave_values[action][0]
        if action in node.local_rave_values:
            beta = self.local_C / (self.local_C + node.num_visits)
            value += beta * node.local_rave_values[action]

        return value

    def selection(self):
        """
            Select the best next node according to UCB function. Finish when node is a leaf
            Returns last node of selection process.
        """

        current_node = self.tree.root

        while not current_node.is_leaf():
            best_next_node = None
            best_ucb_value = float('-inf')
            best_action = None

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
        best_action = None
        best_value = float('-inf')

        for action in self.tree.root.children:
            value = self.tree.root.children[action].value
            if action in self.tree.global_rave_values:
                value += self.tree.global_rave_values[action][0]
            if action in self.tree.root.local_rave_values:
                value += self.tree.root.local_rave_values[action]

            if value > best_value:
                best_action = action
                best_value = value

        if best_action not in self.tree.root.children:
            return None, None

        return self.tree.root.children[best_action], best_action

    def attack_one(self, original_label, tokenized_text):

        original_tokenized_text = tokenized_text
        original_confidence = self._call_model(
            [original_tokenized_text]).squeeze()[original_label].item()
        max_tree_depth = min(self.window_size, len(tokenized_text.words))
        max_words_changed = min(self.max_words_changed, len(tokenized_text.words))

        self.tree = SearchTree(tokenized_text, original_label, original_confidence, max_tree_depth)

        original_root = self.tree.root
        final_action_history = []

        if PROFILE:
            profiler = cProfile.Profile()
            profiler.enable()

        for i in range(max_words_changed):
            best_next_node, best_action = self.run_mcts()
            if best_next_node is None:
                break

            self.tree.root = best_next_node
            self.tree.reset_node_depth()
            final_action_history.append(best_action)
            _, new_label = self.evaluate(self.tree.root)

            if new_label != original_label:
                new_tokenized_text = self.tree.root.text
                break

        if PROFILE:
            profiler.disable()

            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        if LOG_OUPUTS:
            print("Logging...")
            file_name = f"mcts_{time.time()}"
            status = original_label != new_label
            generate_dot(original_root, final_action_history, file_name, status)

            with open(f"outputs/{file_name}_grave.txt", 'w') as f:
                f.write("Original Text: " + original_tokenized_text.text + "\n")
                for action in self.tree.global_rave_values:
                    value, num = self.tree.global_rave_values[action]
                    f.write(f"Action: ({action[0]}, {action[1]})\n")
                    f.write(f"Value: {value}   |   Number actions taken: {num} \n\n")

        if original_label == new_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult(
                original_tokenized_text,
                new_tokenized_text,
                original_label,
                new_label
            )
