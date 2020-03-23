import lru
import numpy as np
import os
import random

from textattack.shared import utils
from textattack.constraints import Constraint
from textattack.shared.tokenized_text import TokenizedText
from textattack.attack_results import AttackResult, FailedAttackResult, SkippedAttackResult

class Attack:
    """
    An attack generates adversarial examples on text. 
    
    This is an abstract class that contains main helper functionality for 
    attacks. An attack is comprised of a search method and a transformation, as 
    well asone or more linguistic constraints that examples must pass to be 
    considered successfully fooling the model.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The transformation applied at each step of the attack.
        constraints: A list of constraints to add to the attack
        is_black_box: Whether or not the attack is black box.

    """
    def __init__(self, goal_function, transformation, constraints=[], is_black_box=True):
        """ Initialize an attack object. Attacks can be run multiple times.
        """
        self.goal_function = goal_function
        if not self.goal_function:
            raise NameError('Cannot instantiate attack without self.goal_function for prediction scores')
        if not hasattr(self, 'tokenizer'):
            if hasattr(self.goal_function.model, 'tokenizer'):
                self.tokenizer = self.goal_function.model.tokenizer
            else:
                raise NameError('Cannot instantiate attack without tokenizer')
        self.transformation = transformation
        self.constraints = constraints
        self.is_black_box = is_black_box
        self.constraints_cache = lru.LRU(2**18)
    
    def get_transformations(self, text, original_text=None, 
                            apply_constraints=True, **kwargs):
        """
        Filters a list of transformations by self.constraints. 
        
        Args:
            transformation: 
            text:
            original text (:obj:`type`, optional): Defaults to None. 
            apply_constraints:
            **kwargs:

        Returns:
            A filtered list of transformations where each transformation matches the constraints

        """
        if not self.transformation:
            raise RuntimeError('Cannot call `get_transformations` without a transformation.')
        transformations = np.array(self.transformation(text, **kwargs))
        if apply_constraints:
            return self._filter_transformations(transformations, text, original_text)
        return transformations
    
    def _filter_transformations_uncached(self, original_transformations, text, original_text=None):
        """ Filters a list of potential perturbations based on a list of
                transformations.
            
            Args:
                transformations (list: function): a list of transformations 
                    that filter a list of candidate perturbations
                text (list: TokenizedText): a list of TokenizedText objects
                    representation potential perturbations
        """
        transformations = original_transformations[:]
        for C in self.constraints:
            if len(transformations) == 0: break
            transformations_mask = C.call_many(text, transformations, original_text)
        # Default to false for all original transformations.
        for original_transformation in original_transformations:
            self.constraints_cache[original_transformation] = False
        # Set unfiltered transformations to True in the cache.
        for successful_transformation in transformations:
            self.constraints_cache[successful_transformation] = True
        return transformations 
     
    def _filter_transformations(self, transformations, text, original_text=None):
        """ Filters a list of potential perturbations based on a list of
                transformations. Checks cache first.
            
            Args:
                transformations (list: function): a list of transformations 
                    that filter a list of candidate perturbations
                text (list: TokenizedText): a list of TokenizedText objects
                    representation potential perturbations
        """
        # Populate cache with transformations.
        uncached_transformations = [t for t in transformations if (t not in self.constraints_cache)]
        self._filter_transformations_uncached(uncached_transformations, text, original_text=original_text)
        # Return transformations from cache.
        return [t for t in transformations if self.constraints_cache[t]]

    def attack_one(self, tokenized_text):
        """
        Perturbs `tokenized_text` to until goal is reached.

        """
        raise NotImplementedError()
 
    def _get_examples_from_dataset(self, dataset, num_examples=None, shuffle=False,
            attack_n=False):
        """ 
        Gets examples from a dataset and tokenizes them.

        Args:
            dataset: An iterable of (label, text) pairs
            num_examples (int): the number of examples to return
            shuffle (:obj:`bool`, optional): Whether to shuffle the data
            attack_n (bool): If true, returns `num_examples` non-skipped
                examples. If false, returns `num_examples` total examples.
        
        Returns:
            results (List[Tuple[Int, TokenizedText, Boolean]]): a list of
                objects containing (label, text, was_skipped)
        """
        examples = []
        n = 0
        for output, text in dataset:
            tokenized_text = TokenizedText(text, self.tokenizer)
            if self.goal_function.should_skip(tokenized_text, output):
                if not attack_n: 
                    n += 1
                examples.append((output, tokenized_text, True))
            else:
                n += 1
                examples.append((output, tokenized_text, False))
            if num_examples is not None and (n >= num_examples):
                break

        if shuffle:
            random.shuffle(examples)
    
        return examples

    def attack_dataset(self, dataset, num_examples=None, shuffle=False, attack_n=False):
        """ 
        Runs an attack on the given dataset and outputs the results to the console and the output file.

        Args:
            dataset: An iterable of (label, text) pairs
            shuffle (:obj:`bool`, optional): Whether to shuffle the data. Defaults to False.
        """
      
        examples = self._get_examples_from_dataset(dataset, 
            num_examples=num_examples, shuffle=shuffle, attack_n=attack_n)

        for output, tokenized_text, was_skipped in examples:
            if was_skipped:
                yield SkippedAttackResult(tokenized_text, output)
                continue
            # Start at 1 since we called once to determine that prediction was correct
            self.goal_function.num_queries = 1
            result = self.attack_one(tokenized_text, output)
            result.num_queries = self.goal_function.num_queries
            yield result
