import difflib
import numpy as np
import os
import torch
import random

from textattack import utils as utils

from textattack.constraints import Constraint
from textattack.tokenized_text import TokenizedText

class Attack:
    """
    An attack generates adversarial examples on text. 

    Args:
        model: A PyTorch or TensorFlow model to attack
        constraints: A list of constraints to add to the attack

    """
    def __init__(self, model, constraints=[]):
        """ Initialize an attack object.
        
        Attacks can be run multiple times
        
         @TODO should `tokenizer` be an additional parameter or should
            we assume every model has a .tokenizer ?
        """
        self.model = model
        # Transformation and corresponding constraints.
        self.constraints = []
        if constraints:
            self.add_constraints(constraints)
        # List of files to output to.
        self.output_files = []
        self.output_to_terminal = True
        self.output_to_visdom = False
    
    def add_output_file(self, file):
        """ 
        When attack runs, it will output to this file. 

        Args:
            file (str): The path to the output file
            
        """
        if isinstance(file, str):
            directory = os.path.dirname(file)
            if not os.path.exists(directory):
                os.makedirs(directory)
            file = open(file, 'w')
        self.output_files.append(file)
        
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

    def _attack_one(self, label, tokenized_text):
        """
        Perturbs `text` to until `self.model` gives a different label
        than `label`. 

        """
        raise NotImplementedError()
      
    def _call_model(self, tokenized_text_list):
        """
        Returns model predictions for a list of TokenizedText objects. 
        
        """
        ids = torch.tensor([t.ids for t in tokenized_text_list])
        ids = ids.to(utils.get_device())
        scores = self.model(ids)
        return scores
      
    def attack(self, dataset, shuffle=False):
        """ 
        Runs an attack on the given dataset and outputs the results to the console and the output file.

        Args:
            dataset: An iterable of (label, text) pairs
            shuffle (:obj:`bool`, optional): Whether to shuffle the data. Defaults to False.

        Returns:
            The results of the attack on the dataset

        """
        if shuffle:
            random.shuffle(dataset)
        
        results = []
        for label, text in dataset:
            tokenized_text = TokenizedText(self.model, text)
            result = self._attack_one(label, tokenized_text)
            results.append(result)
        
        if self.output_to_terminal:
            for i, result in enumerate(results):
                print('-'*35, 'Result', str(i+1), '-'*35)
                result.print_()
                print()
        
        if self.output_files:
            for output_file in self.output_files:
                for result in results:
                    output_file.write(str(result) + '\n')
        
        if self.output_to_visdom:
            raise NotImplementedError()
        
        print('-'*80)
        
        return results

class AttackResult:
    """
    Result of an Attack run on a single (label, text_input) pair. 

    Args:
        original_text (str): The original text
        perturbed_text (str): The perturbed text resulting from the attack
        original_label (int): he classification label of the original text
        perturbed_label (int): The classification label of the perturbed text

    """
    def __init__(self, original_text, perturbed_text, original_label,
        perturbed_label):
        if original_text is None:
            raise ValueError('Attack original text cannot be None')
        if perturbed_text is None:
            raise ValueError('Attack perturbed text cannot be None')
        if original_label is None:
            raise ValueError('Attack original label cannot be None')
        if perturbed_label is None:
            raise ValueError('Attack perturbed label cannot be None')
        self.original_text = original_text
        self.perturbed_text = perturbed_text
        self.original_label = original_label
        self.perturbed_label = perturbed_label
    
    def __data__(self):
        data = (self.original_text, self.original_label, self.perturbed_text,
                self.perturbed_label)
        return tuple(map(str, data))
    
    def __str__(self):
        return '\n'.join(self.__data__())
    
    def diff_color(self):
        """ 
        Highlights the difference between two texts using color.
        
        """
        _color = utils.color_text_terminal
        t1 = self.original_text
        t2 = self.perturbed_text
        
        words1 = t1.words()
        words2 = t2.words()
        
        c1 = utils.color_from_label(self.original_label)
        c2 = utils.color_from_label(self.perturbed_label)
        new_is = []
        new_w1s = []
        new_w2s = []
        for i in range(min(len(words1), len(words2))):
            w1 = words1[i]
            w2 = words2[i]
            if w1 != w2:
                new_is.append(i)
                new_w1s.append(_color(w1, c1))
                new_w2s.append(_color(w2, c2))
        
        t1 = self.original_text.replace_words_at_indices(new_is, new_w1s)
        t2 = self.original_text.replace_words_at_indices(new_is, new_w2s)
                
        return (str(t1), str(t2))
    
    def print_(self):
        print(str(self.original_label), '-->', str(self.perturbed_label))
        print('\n'.join(self.diff_color()))

class FailedAttackResult(AttackResult):
    def __init__(self, original_text, original_label):
        if original_text is None:
            raise ValueError('Attack original text cannot be None')
        if original_label is None:
            raise ValueError('Attack original label cannot be None')
        self.original_text = original_text
        self.original_label = original_label

    def __data__(self):
        data = (self.original_text, self.original_label)
        return tuple(map(str, data))

    def print_(self):
        _color = utils.color_text_terminal
        print(str(self.original_label), '-->', _color('[FAILED]', 'red'))
        print(self.original_text)

if __name__ == '__main__':
    import time
    import socket
    
    import textattack.attacks as attacks
    import textattack.constraints as constraints
    from textattack.datasets import YelpSentiment
    from textattack.models import BertForSentimentClassification
    from textattack.transformations import WordSwapEmbedding
    
    start_time = time.time()
    
    def __data__(self):
        data = (self.original_text, self.original_label)
        return tuple(map(str, data))
    
    def print_(self):
        _color = utils.color_text_terminal
        print(str(self.original_label), '-->', _color('[FAILED]', 'red'))
        print(self.original_text)
