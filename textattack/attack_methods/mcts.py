from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.blackbox import BlackBoxAttack
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Reward functions for MCTS
def raw_prob_reward(prob, orig_label, new_label="None"):
    # New_label is only set for targetted attacks
    prob_exp = torch.exp(output)
    v1 = prob_exp.data[original_label].clone() 
    prob_exp.data[original_label] = 0 
    v2 = prob_exp.max().data
    return (v2 - v1).item()

def entropy_reward(orig_label, prob):
    return F.nll_loss(prob, orig_label).data.cpu()[0].item()

class Node:
    """ Helper node class used in implementation of MCTS"""
    def __init__(self,feature_set):
        self.feature_set = feature_set  #Represents State
        self.f_size = self.feature_set.sum()  #Represents number of positive words for transofrmation
        self.childrens = {} 
        self.T_f = .0
        self.av = .0
        self.allowed_features = feature_set.nonzero()
        self.lrave_count = np.array(feature_set.shape)
        self.lrave_reward = np.array(feature_set.shape).astype(float)
        self.lrave_variance = np.array(feature_set.shape).astype(float)
        self.lrave_score = np.array(feature_set.shape).astype(float)

class Tree:
    """ Helper tree class used in implementation of MCTS; dictionary of nodes"""
    def __init__(self,nsize):
        self.tree = {}
        root = np.array([False] * nsize,dtype=bool)
        self.tree[root.tobytes()] = Node(root) #???

    def find(self,feature_set):
        """ Returns node containing features_set if present in tree"""
        if feature_set.tobytes() in self.tree:
            return self.tree[feature_set.tobytes()]
        else:
            return None

    def save(self,feature_set,node):
        """ Adds node with feature_set to tree"""
        self.tree[feature_set.tobytes()] = node

class MCTS():
    """ 
    Uses Monte Carlo Tree Search (MCTS) to attempt to find the most important words in an input.
    Args:
        model: A PyTorch or TensorFlow model to attack.
        transformation: The type of transformation to use. Should be a subclass of WordSwap. 
        constraints: A list of constraints to add to the attack
        reward_type (str): Defines what type of function to use for MCTS.
            - raw_prob: Uses "max_{c'} P(c'|s) - P(c_l|s)"
            - entropy: Uses negative log-likelihood function
        max_iter (int) : Maximum iterations for MCTS. Default is 4000
        max_words_changed (int) : Maximum number of words we change during MCTS. Effectively represents depth of search tree.
    """
    def __init__(self, model, transformation, constraints=[], 
        reward_type="raw_prob", max_iter=4000, max_words_changed=10
    ):
        super().__init__(model, transformation, constraints=constraints)
        self.reward_type = reward_type
        self.max_iter = max_iter
        self.max_words_changed = max_words_changed
        self.alltimebest = -1e9
        self.bestfeature = []

        if reward_type == "raw_prob":
            self.reward_func = raw_prob_reward 
        elif reward_type == "entropy":
            self.reward_func = entropy_reward

    def get_reward(self, current_state, input_text):
        for i in range(len(current_state)):
            if current_state[i]:
                transformed_candidates = self.get_transformations(
                                            test_input,
                                            indices_to_replace=[k]
                                        )

                if len(transformed_candidates) > 0:
                    rand = np.random.randint(len(transformed_candidates))
                    transformed_text = transformed_candidates[rand]

        #Evaluate current features against model
        output = self._call_model([transformed_text])[0]

    def UCB(self):
        pass

    def expansion(self):
        pass

    def simulation(self):
        pass

    def back_up(self):
        pass

    def selection(self, current_state, depth):
        if depth>=self.params.max_depth:
            reward_value = self.reward_func(current_state)
            self.update_gRAVE(current_state, reward_V)
        else:
            next_node = self.update_Tree_And_Get_Address(current_state)

            if (next_node.T_f != 0):
                fi = self.UCB(next_node)
                if (fi == -1): # it means that no feature has been selected and that we are going to perform random exploration
                    depth = current_state.sum()
                    reward_V = self.iterate_random(self.tree, current_state)
                    self.update_gRAVE(current_state, reward_V)
                else: #add the feature to the feature set
                    current_state[fi] = True
                    reward_V = self.iterate(current_state, depth+1)
            else:
                depth_now = current_state.sum()
                reward_V = self.iterate_random(self.tree, current_state)
                self.update_gRAVE(current_state, reward_V)
                fi = -1 # indicate that random exploration has been performed and thus no feature selected
            self.update_Node(next_node, fi, reward_V)
        return reward_V


    def run_mcts(self, orig_label, tokenized_input):
        input_size = len(tokenized_input)
        tree = Tree(input_size)
        for i in range(self.max_iter):
            if i % 100 == 0:
                print(f"Running MCTS iteration {i}")
            current_state = np.array([False] * input_size, dtype = bool)
            
            self.selection(current_state, 0)
            self.expansion()
            self.simulation()
            self.back_up()

    def _attack_one(self, original_label, tokenized_text):

        self.runmcts(original_label, tokenized_test.words)

        new_tokenized_text = tokenized_text

        #Transform each index selected by MCTS using given transformation
        indices = []
        for k in range(len(self.bestfeature)):
            if self.bestfeature[k]:
                transformed_text_candidates = self.get_transformations(
                    self.transformation,
                    new_tokenized_text,
                    indices_to_replace=[k])
                if len(transformed_text_candidates) > 0:
                    rand = np.random.randint(len(transformed_text_candidates))
                    new_tokenized_text = transformed_text_candidates[rand]

        new_output = self._call_model([new_tokenized_text])[0]
        new_text_label = self._call_model([new_tokenized_text])[0].argmax().item()

        if original_label == new_text_label:
            return FailedAttackResult(tokenized_text, original_label)
        else:
            return AttackResult( 
                tokenized_text, 
                new_tokenized_text, 
                original_label,
                new_text_label
            )