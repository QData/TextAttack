import difflib
import math
import numpy as np
import os
import torch
import random
import time

from textattack import utils as utils
from textattack.constraints import Constraint
from textattack.tokenized_text import TokenizedText
from textattack.attacks.attack_logger import AttackLogger
from textattack.attacks import AttackResult, FailedAttackResult

class Attack:
    """
    An attack generates adversarial examples on text. 

    Args:
        model: A PyTorch or TensorFlow model to attack
        constraints: A list of constraints to add to the attack

    """
    def __init__(self, constraints=[], is_black_box=True):
        """ Initialize an attack object.
        
        Attacks can be run multiple times
        
         @TODO should `tokenizer` be an additional parameter or should
            we assume every model has a .tokenizer ?
        """
        if not self.model:
            raise NameError('Cannot instantiate attack without self.model for prediction scores')
        if not self.text_to_ids_converter:
            raise NameError('Cannot instantiate attack without tokenizer')
        # Transformation and corresponding constraints.
        self.constraints = []
        if constraints:
            self.add_constraints(constraints)
        # Logger
        self.logger = AttackLogger()
        self.is_black_box = is_black_box
    
    def add_output_file(self, filename):
        """ 
        When attack runs, it will output to this file. 

        Args:
            file (str): The path to the output file
            
        """
        self.logger.add_output_file(filename)
        
    def add_constraint(self, constraint):
        """ 
        Adds a constraint to the attack. 
        
        Args:
            constraint: A constraint to add, see constraints

        Raises:
            ValueError: If the constraint is not of type :obj:`Constraint`

        """
        if not isinstance(constraint, Constraint):
            raise ValueError('Cannot add constraint of type', type(constraint))
        self.constraints.append(constraint)
    
    def add_constraints(self, constraints):
        """ 
        Adds multiple constraints to the attack. 
        
        Args:
            constraints: An iterable of constraints to add, see constraints. 

        Raises:
            TypeError: If the constraints are not iterable

        """
        # Make sure constraints are iterable.
        try:
            iter(constraints)
        except TypeError as te:
            raise TypeError(f'Constraint list type {type(constraints)} is not iterable.')
        # Store each constraint after validating its type.
        for constraint in constraints:
            self.add_constraint(constraint)
    
    def get_transformations(self, transformation, text, original_text=None, 
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
        transformations = np.array(transformation(text, **kwargs))
        if apply_constraints:
            return self._filter_transformations(transformations, text, original_text)
        return transformations
     
    def _filter_transformations(self, transformations, text, original_text=None):
        for C in self.constraints:
            transformations = C.call_many(text, transformations, original_text)
        return transformations 
        
    def enable_visdom(self):
        self.logger.enable_visdom()

    def enable_stdout(self):
        self.logger.enable_stdout()

    def _attack_one(self, label, tokenized_text):
        """
        Perturbs `text` to until `self.model` gives a different label than 
        `label`.

        """
        raise NotImplementedError()
        
    def _call_model(self, tokenized_text_list, batch_size=64):
        """
        Returns model predictions for a list of TokenizedText objects. 
        
        """
        if not len(tokenized_text_list):
            return torch.tensor([])
        self.num_queries += len(tokenized_text_list)
        ids = torch.tensor([t.ids for t in tokenized_text_list])
        num_batches = int(math.ceil(len(tokenized_text_list) / float(batch_size)))
        scores = []
        for batch_i in range(num_batches):
            batch_start = batch_i * batch_size
            batch_stop  = (batch_i + 1) * batch_size
            batch_ids = ids[batch_start:batch_stop, :].to(utils.get_device())
            scores.append(self.model(batch_ids))
            del batch_ids
        del ids
        scores = torch.cat(scores, dim=0)
        # Validation check on model score dimensions
        if scores.ndim == 1:
            # Unsqueeze prediction, if it's been squeezed by the model.
            if len(tokenized_text_list == 1):
                scores = scores.unsqueeze(dim=0)
            else:
                raise ValueError(f'Model return score of shape {scores.shape} for {len(tokenized_text_list)} inputs.')
        elif scores.ndim != 2:
            # If model somehow returns too may dimensions, throw an error.
            raise ValueError(f'Model return score of shape {scores.shape} for {len(tokenized_text_list)} inputs.')
        elif scores.shape[0] != len(tokenized_text_list):
            # If model returns an incorrect number of scores, throw an error.
            raise ValueError(f'Model return score of shape {scores.shape} for {len(tokenized_text_list)} inputs.')
        elif not ((scores.sum(dim=1) - 1).abs() < 1e-6).all():
            # Values in each row should sum up to 1. The model should return a 
            # set of numbers corresponding to probabilities, which should add
            # up to 1. Since they are `torch.float` values, allow a small
            # error in the summation.
            raise ValueError('Model scores do not add up to 1.')
        return scores
      
    def attack(self, dataset, shuffle=False):
        """ 
        Runs an attack on the given dataset and outputs the results to the console and the output file.

        Args:
            dataset: An iterable of (label, text) pairs
            shuffle (:obj:`bool`, optional): Whether to shuffle the data. Defaults to False.
        """
    
        self.logger.start_time = time.time()
        self.logger.log_attack_details(self.__class__.__name__, 
                                       self.is_black_box,    
                                       self.model_description)
        
        if shuffle:
            random.shuffle(dataset)
        
        for label, text in dataset:
            self.num_queries = 0
            tokenized_text = TokenizedText(text, self.text_to_ids_converter)
            predicted_label = self._call_model([tokenized_text])[0].argmax().item()
            if predicted_label != label:
                self.logger.log_skipped(tokenized_text)
                continue
            result = self._attack_one(label, tokenized_text)
            result.num_queries = self.num_queries
            self.logger.log_result(result)
        
        self.logger.log_sep()
        self.logger.log_summary(self.is_black_box)
        self.logger.flush()
