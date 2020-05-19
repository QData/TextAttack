import lru
import numpy as np
import os
import random

from textattack.shared import utils
from textattack.constraints import Constraint, PreTransformationConstraint
from textattack.shared import TokenizedText
from textattack.attack_results import SkippedAttackResult, SuccessfulAttackResult, FailedAttackResult

class Attack:
    """
    An attack generates adversarial examples on text. 
    
    This is an abstract class that contains main helper functionality for 
    attacks. An attack is comprised of a search method, goal function, 
    a transformation, and a set of one or more linguistic constraints that 
    successful examples must meet.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        constraints: A list of constraints to add to the attack, defining which perturbations are valid.
        transformation: The transformation applied at each step of the attack.
        search_method: A strategy for exploring the search space of possible perturbations
        is_black_box: Whether or not the attack is black box.

    """

    def __init__(self, goal_function=None, constraints=[], transformation=None, search_method=None):
        """ Initialize an attack object. Attacks can be run multiple times.
        """
        self.search_method = search_method
        self.goal_function = goal_function
        if not self.goal_function:
            raise NameError('Cannot instantiate attack without self.goal_function for predictions')
        if not hasattr(self, 'tokenizer'):
            if hasattr(self.goal_function.model, 'tokenizer'):
                self.tokenizer = self.goal_function.model.tokenizer
            else:
                raise NameError('Cannot instantiate attack without tokenizer')
        self.transformation = transformation
        self.is_black_box = getattr(transformation, 'is_black_box', True)

        if not self.search_method.check_transformation_compatibility(self.transformation):
            raise ValueError('SearchMethod {self.search_method} incompatible with transformation {self.transformation}')

        self.constraints = []
        self.pre_transformation_constraints = []
        for constraint in constraints:
            if isinstance(constraint, PreTransformationConstraint):
                self.pre_transformation_constraints.append(constraint)
            else:
                self.constraints.append(constraint)
        
        self.constraints_cache = lru.LRU(utils.config('CONSTRAINT_CACHE_SIZE'))
        
        # Give search method access to functions for getting transformations and evaluating them
        self.search_method.get_transformations = self.get_transformations
        self.search_method.get_goal_results = self.goal_function.get_results 
    
    def get_transformations(self, text, original_text=None, 
                            apply_constraints=True, **kwargs):
        """
        Filters a list of transformations by self.constraints. 
        
        Args:
            transformation: 
            text:
            original text (:obj:`type`, optional): Defaults to None. 
            apply_constraints: Whether or not to apply post-transformation constraints
            **kwargs:

        Returns:
            A filtered list of transformations where each transformation matches the constraints

        """
        if not self.transformation:
            raise RuntimeError('Cannot call `get_transformations` without a transformation.')
       
        transformations = np.array(self.transformation(text, 
                            pre_transformation_constraints=self.pre_transformation_constraints, 
                            **kwargs))
        if apply_constraints:
            return self._filter_transformations(transformations, text, original_text)
        return transformations
    
    def _filter_transformations_uncached(self, original_transformations, text, original_text=None):
        """ Filters a list of potential perturbations based on a list of
                transformations. Checks cache first.
            
            Args:
                transformations (list: function): a list of transformations 
                    that filter a list of candidate perturbations
                text (list: TokenizedText): a list of TokenizedText objects
                    representation potential perturbations
        """
        transformations = original_transformations[:]
        for C in self.constraints:
            if len(transformations) == 0: break
            transformations = C.call_many(text, transformations, original_text=original_text)
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
        uncached_transformations = []
        for t in transformations:
            if t not in self.constraints_cache:
                uncached_transformations.append(t)
            else:
                # promote t to the top of the LRU cache
                self.constraints_cache[t] = self.constraints_cache[t]
        self._filter_transformations_uncached(uncached_transformations, text, original_text=original_text)
        # Return transformations from cache.
        filtered_transformations = [t for t in transformations if self.constraints_cache[t]]
        # Sort transformations to ensure order is preserved between runs.
        filtered_transformations.sort(key=lambda t: t.text)
        return filtered_transformations

    def attack_one(self, initial_result):
        """
        Perturbs `tokenized_text` from initial_result until goal is reached.
        """
        final_result = self.search_method(initial_result)
        if final_result.succeeded:
            return SuccessfulAttackResult(initial_result, final_result)
        else:
            return FailedAttackResult(initial_result, final_result)
 
    def _get_examples_from_dataset(self, dataset, num_examples=None, shuffle=False,
            attack_n=False, attack_skippable_examples=False):
        """ 
        Gets examples from a dataset and tokenizes them.

        Args:
            dataset: An iterable of (text, ground_truth_output) pairs
            num_examples (int): the number of examples to return
            shuffle (:obj:`bool`, optional): Whether to shuffle the data
            attack_n (bool): If `True`, returns `num_examples` non-skipped
                examples. If `False`, returns `num_examples` total examples.
        
        Returns:
            results (Iterable[Tuple[GoalFunctionResult, Boolean]]): a list of
                objects containing (text, ground_truth_output, was_skipped)
        """
        examples = []
        n = 0
        
        if shuffle:
            random.shuffle(dataset.examples)

        if num_examples <= 0:
            return
            yield
            
        for text, ground_truth_output in dataset:
            tokenized_text = TokenizedText(text, self.tokenizer)
            goal_function_result = self.goal_function.get_result(tokenized_text, ground_truth_output)
            # We can skip examples for which the goal is already succeeded,
            # unless `attack_skippable_examples` is True.
            if (not attack_skippable_examples) and (goal_function_result.succeeded):
                if not attack_n: 
                    n += 1
                # Store the true output on the goal function so that the
                # SkippedAttackResult has the correct output, not the incorrect.
                goal_function_result.output = ground_truth_output
                yield (goal_function_result, True)
            else:
                n += 1
                yield (goal_function_result, False)
            if num_examples is not None and (n >= num_examples):
                break

    def attack_dataset(self, dataset, num_examples=None, shuffle=False, attack_n=False):
        """ 
        Runs an attack on the given dataset and outputs the results to the 
            console and the output file.

        Args:
            dataset: An iterable of (text, ground_truth_output) pairs
            shuffle (:obj:`bool`, optional): Whether to shuffle the data. Defaults to False.
        """
        
        examples = self._get_examples_from_dataset(dataset, 
            num_examples=num_examples, shuffle=shuffle, attack_n=attack_n)

        for goal_function_result, was_skipped in examples:
            if was_skipped:
                yield SkippedAttackResult(goal_function_result)
                continue
            # Start query count at 1 since we made a single query to determine 
            # that the prediction was correct.
            self.goal_function.num_queries = 1
            result = self.attack_one(goal_function_result)
            result.num_queries = self.goal_function.num_queries
            yield result
    
    def __repr__(self):
        """ Prints attack parameters in a human-readable string.
            
        Inspired by the readability of printing PyTorch nn.Modules:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
        """
        main_str = 'Attack' + '('
        lines = []
       
        lines.append(
            utils.add_indent(f'(search_method): {self.search_method}', 2)
        )
        # self.goal_function
        lines.append(
            utils.add_indent(f'(goal_function):  {self.goal_function}', 2)
        )
        # self.transformation
        lines.append(
            utils.add_indent(f'(transformation):  {self.transformation}', 2)
        )
        # self.constraints
        constraints_lines = []
        constraints = self.constraints + self.pre_transformation_constraints
        if len(constraints):
            for i, constraint in enumerate(constraints):
                constraints_lines.append(utils.add_indent(f'({i}): {constraint}', 2))
            constraints_str = utils.add_indent('\n' + '\n'.join(constraints_lines), 2)
        else:
            constraints_str = 'None'
        lines.append(utils.add_indent(f'(constraints): {constraints_str}', 2))
        # self.is_black_box
        lines.append(utils.add_indent(f'(is_black_box):  {self.is_black_box}', 2))
        main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
    
    __str__ = __repr__
