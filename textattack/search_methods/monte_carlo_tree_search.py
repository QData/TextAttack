from .attack import Attack
from textattack.attack_results import SuccessfulAttackResult, FailedAttackResult
import math
import random
import statistics
import collections

class Node:
    """
        Represents a node in search tree
        Attributes:
            id (int): unique int for id
            text (TokenizedText) : Version of TokenizedText that Node represents
            parent (Node): Parent node
            depth (int): Current depth in search tree
            num_visits (int): Number of visits to the current node
            value (float): Score of adversarial attack
            local_rave_values ((int, str) --> value (float)): Store local RAVE value
            value_history (list): Stores the history of score/reward gained when choosing this node at every iteration.
                                    Used for calculating variance.
            variance (float): Variance of score across trials
            children ((int, str) --> Node): Map action to child Node
            ...
    """

    def __init__(self, text, parent):
        self.text = text
        self.parent = parent
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
        self.num_visits = 0
        self.value = 0.0
        self.local_rave_values = {}
        self.value_history = []
        self.children = {}

    def is_leaf(self):
        return not bool(self.children)

    @property
    def variance(self):
        if len(self.value_history) < 2:
            return 0
        else:
            return statistics.variance(self.value_history, self.value)

class SearchTree:
    """
        Tree used to hold states/variables for MCTS.
        Each node represents a particular TokenizedText that reflects certain transformation.
        Each action is represented as tuple of (int, str) where int is the i-th word chosen for transformation
        and str is the specific word that i-th word is transformed to.

        root (Node): root of search tree
        original_text (TokenizedText): TokenizedText that is under attack
        original_label (int)
        max_depth (int): max depth of search tree
    """

    NOOP_ACTION = (-1, '<noop>')

    def __init__(self, original_text, original_label, max_depth):
        self.root = Node(original_text, None)
        self.original_text = original_text
        self.original_label = original_label
        self.max_depth = max_depth

        self.available_words_to_transform = set(range(len(self.original_text.words)))
        self.words_to_skip = set()
        self.action_history = []
        self.iteration = 0 
        self.global_rave_values = {}

    def clear_single_iteration_history(self):
        self.available_words_to_transform = set(range(len(self.original_text.words)))
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
        num_iter (int): Number of iterations for MCTS.
        top_k (int): Top-k transformations to select when expanding search tree.
        max_words_changed (int) : Maximum number of words we change during MCTS. Effectively represents depth of search tree.
    """

    def __init__(self, model, transformation, constraints=[], num_iter=100, max_words_changed=32):
        super().__init__(model, transformation, constraints=constraints)

        # MCTS Hyper-parameters
        self.num_iter = num_iter
        self.max_words_changed = max_words_changed
        self.max_tree_depth = 8
        self.step_size = 2
        self.ucb_C = 2
        self.global_C = 30
        self.local_C = 2

    def _backprop(self, current_node, search_value):
        """
            Update score statistics for each node, starting from last leaf node of search iteration
                and ending at the root.
            Also update global RAVE values. No return value.
        """

        # Update global RAVE values
        for action in self.search_tree.action_history:
            if action in self.search_tree.global_rave_values:
                old_rave = self.search_tree.global_rave_values[action]
                new_value = (old_rave[0] * old_rave[1] + search_value) / (old_rave[1] + 1)

                self.search_tree.global_rave_values[action] = (new_value, old_rave[1] + 1)
            else:
                self.search_tree.global_rave_values[action] = (search_value, 1)

        while current_node is not None:
            n = current_node.num_visits
            current_node.num_visits += 1
            current_node.value = (current_node.value * n + search_value) / (n + 1)
            current_node.value_history.append(search_value)

            # Update local RAVE values
            for action in self.search_tree.action_history:
                if action in current_node.local_rave_values:
                    current_node.local_rave_values[action] = (current_node.local_rave_values[action] \
                        * n + search_value) / (n + 1)
                else:
                    current_node.local_rave_values[action] = search_value

            current_node = current_node.parent

    def _expansion(self, current_node):
        """
            Create next nodes based on available transformations and then take a random action.
            Returns: New node that we expand to. If no such node exists, return None
        """
        available_transformations = []
        available_words_to_transform = list(self.search_tree.available_words_to_transform)
        while len(available_transformations) == 0 and available_words_to_transform:
            # Randomly sample one word and find transformations.
            word_to_transform = random.choice(available_words_to_transform)
            available_transformations = self.get_transformations(
                current_node.text,
                original_text=self.search_tree.original_text,
                indices_to_replace=[word_to_transform]
            )
            available_words_to_transform.remove(word_to_transform)

        if len(available_words_to_transform) == 0:
            # No transformations available
            return current_node
        else:
            available_actions = [(t.attack_attrs['modified_word_index'],
                                  t.attack_attrs['new_word']) for t in available_transformations]

            for i in range(len(available_actions)):
                current_node.children[available_actions[i]] = Node(
                    available_transformations[i],
                    current_node
                )
            # Add no-op node
            current_node.children[SearchTree.NOOP_ACTION] = Node(current_node.text, current_node)

            random_action = random.choice(available_actions)
            if random_action != SearchTree.NOOP_ACTION:
                self.search_tree.available_words_to_transform.remove(word_to_transform)
            self.search_tree.action_history.append(random_action)

            return current_node.children[random_action]

    def _UCB(self, node, action):
        return node.children[action].value + math.sqrt(
            self.ucb_C * math.log(node.num_visits) / node.children[action].num_visits
        )

    def _UCB_tuned(self, node, action):
        return  node.children[action].value + math.sqrt(
            self.ucb_C * math.log(node.num_visits) / node.children[action].num_visits
            * min(0.25, node.children[action].variance + math.sqrt(2 * math.log(node.num_visits) / node.children[action].num_visits))
        )

    def _UCB_RAVE_tuned(self, node, action):
        ucb = math.sqrt(
            self.ucb_C * math.log(node.num_visits) / max(1, node.children[action].num_visits)
            * min(0.25, node.children[action].variance
            + math.sqrt(2 * math.log(node.num_visits) / max(1, node.children[action].num_visits)))
        )
        global_rave = 0.0
        beta = 0.0
        if action in self.search_tree.global_rave_values:
            global_rave = self.search_tree.global_rave_values[action][0]
            beta = self.global_C / (self.global_C + self.search_tree.global_rave_values[action][1])

        #print(f"{node.children[action].value} | {global_rave} | {ucb}")
        return (1 - beta) * node.children[action].value + beta * global_rave + ucb

    def _selection(self):
        """
            Select the best next node according to UCB function. Finish when node is a leaf
            Returns last node of selection process.
        """

        current_node = self.search_tree.root

        while not current_node.is_leaf():
            best_next_node = None
            best_ucb_value = float('-inf')
            best_action = None

            for action in current_node.children.keys():
                ucb_value = self._UCB_RAVE_tuned(
                    current_node, 
                    action
                )

                if ucb_value > best_ucb_value:
                    best_next_node = current_node.children[action]
                    best_ucb_value = ucb_value
                    best_action = action

            current_node = best_next_node
            if best_action != SearchTree.NOOP_ACTION:
                self.search_tree.available_words_to_transform.remove(best_action[0])
            self.search_tree.action_history.append(best_action)

        return current_node

    def _run_mcts(self, num_iter):
        """
            Runs Monte Carlo Tree Search at the current root.
            Returns best node and best action.
        """

        for i in range(num_iter):
            #print(f'Iteration {i+1}')
            self.search_tree.iteration += 1
            self.search_tree.clear_single_iteration_history()
            current_node = self._selection()

            previous_node = None
            while previous_node != current_node and current_node.depth < self.search_tree.max_depth:
                previous_node = current_node
                current_node = self._expansion(current_node)

            result = self.goal_function.get_results([current_node.text], self.search_tree.original_label)[0]
            search_value = 1 + result.score if result.score < 0 else result.score
            self._backprop(current_node, search_value)

    def _choose_best_move(self, node):
        best_action = None
        best_value = float('-inf')
        for action in node.children:
            value = node.children[action].value
            global_rave = 0.0
            beta = 0.0
            if action in self.search_tree.global_rave_values:
                global_rave += self.search_tree.global_rave_values[action][0]
                beta = self.global_C / (self.global_C + self.search_tree.global_rave_values[action][1])
            value = (1 - beta) * value + beta * global_rave     

            if value > best_value:
                best_action = action
                best_value = value

        if best_action not in node.children:
            return None, None
        else:
            return node.children[best_action], best_action

    def attack_one(self, tokenized_text, correct_output):

        original_result = self.goal_function.get_results([tokenized_text], correct_output)[0]
        max_tree_depth = min(self.max_tree_depth, len(tokenized_text.words))
        max_words_changed = min(self.max_words_changed, len(tokenized_text.words))
        num_iter = self.num_iter

        self.search_tree = SearchTree(tokenized_text, original_result.output, max_tree_depth)
        current_result = original_result
        words_changed = 0

        while words_changed < max_words_changed:
            self._run_mcts(num_iter)

            root = self.search_tree.root
            i = 0
            while i < self.step_size:
                root, action = self._choose_best_move(root)
                if not root:
                    break
                
                current_result = self.goal_function.get_results([root.text], self.search_tree.original_label)[0]
                if current_result.output != correct_output:
                    break

                if action != SearchTree.NOOP_ACTION:
                    words_changed += 1
                i+=1
            
            if not root:
                break
            if current_result.output != correct_output:
                break

            self.search_tree.root = root
            self.search_tree.root.parent = None
            self.search_tree.reset_node_depth()
            num_iter = int(num_iter * 0.9)

        if correct_output == current_result.output:
            return FailedAttackResult(original_result, current_result)
        else:
             return SuccessfulAttackResult(original_result, current_result)