from textattack.attacks import AttackResult, FailedAttackResult
from textattack.attacks.blackbox import BlackBoxAttack
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

class MCTS(Attack):
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
        reward_type="prob_diff", max_iter=4000, max_words_changed=10
    ):
        super().__init__(model, transformation, constraints=constraints)
        self.reward_type = reward_type
        self.max_iter = max_iter
        self.max_words_changed = max_words_changed
        self.alltimebest = -1e9
        self.bestfeature = []

        if reward_type == "prob_diff":
            self.reward_value = prob_diff_value
        elif reward_type == "entropy":
            self.reward_value = entropy_value

        self.tree = None
        self.ucb_C = 2

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


    def attack_one(self, original_label, tokenized_text):

        original_tokenized_text = tokenized_text
        original_prob = self._call_model([tokenized_text]).squeeze().max()
        num_words_changed = 0
        unswapped_word_indices = list(range(len(tokenized_text.words)))
        new_tokenized_text = None
        new_text_label = None

        print(tokenized_text)

        transformed_text_candidates = self.get_transformations(
                tokenized_text,
                indices_to_replace=unswapped_word_indices)

        if original_label == new_text_label:
            return FailedAttackResult(original_tokenized_text, original_label)
        else:
            return AttackResult( 
                original_tokenized_text, 
                new_tokenized_text, 
                original_label,
                new_text_label
            )