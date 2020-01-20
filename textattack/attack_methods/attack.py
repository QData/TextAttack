import math
import numpy as np
import os
import torch
import random
import time

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
        model: A PyTorch or TensorFlow model to attack
        constraints: A list of constraints to add to the attack

    """
    def __init__(self, model, transformation, constraints=[], is_black_box=True):
        """ Initialize an attack object. Attacks can be run multiple times.
        """
        self.model = model
        self.model_description = model.__class__.__name__
        if not self.model:
            raise NameError('Cannot instantiate attack without self.model for prediction scores')
        if not hasattr(self, 'tokenizer'):
            if hasattr(self.model, 'tokenizer'):
                self.tokenizer = self.model.tokenizer
            else:
                raise NameError('Cannot instantiate attack without tokenizer')
        self.transformation = transformation
        self.constraints = constraints
        self.is_black_box = is_black_box
    
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
     
    def _filter_transformations(self, transformations, text, original_text=None):
        """ Filters a list of potential perturbations based on a list of
                transformations.
            
            Args:
                transformations (list: function): a list of transformations 
                    that filter a list of candidate perturbations
                text (list: TokenizedText): a list of TokenizedText objects
                    representation potential perturbations
        """
        for C in self.constraints:
            if len(transformations) == 0: break
            transformations = C.call_many(text, transformations, original_text)
        return transformations 

    def attack_one(self, label, tokenized_text):
        """
        Perturbs `text` to until `self.model` gives a different label than 
        `label`.

        """
        raise NotImplementedError()
        
    def _call_model(self, tokenized_text_list, batch_size=8):
        """
        Returns model predictions for a list of TokenizedText objects. 
        
        """
        if not len(tokenized_text_list):
            return torch.tensor([])
        try:
            self.num_queries += len(tokenized_text_list)
        except AttributeError:
            # If some outside class is just using the attack for its `call_model`
            # function, then `self.num_queries` will not have been initialized.
            # In this case, just continue.
            pass
        ids = [t.ids for t in tokenized_text_list]
        ids = torch.tensor(ids).to(utils.get_device()) 
        #
        # shape of `ids` is (n, m, d)
        #   - n: number of elements in `tokenized_text_list`
        #   - m: number of vectors per element
        #           ex: most classification models take a single vector, so m=1
        #           ex: some entailment models take three vectors, so m=3
        #   - d: dimensionality of each vector
        #           (a typical model might set d=128 or d=256)
        num_fields = ids.shape[1]
        num_batches = int(math.ceil(len(tokenized_text_list) / float(batch_size)))
        scores = []
        for batch_i in range(num_batches):
            batch_start = batch_i * batch_size
            batch_stop  = (batch_i + 1) * batch_size
            batch_ids = ids[batch_start:batch_stop]
            batch = [batch_ids[:, x, :] for x in range(num_fields)]
            with torch.no_grad():
                preds = self.model(*batch)
            scores.append(preds)
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
 
    def _get_examples_from_dataset(self, dataset, num_examples=None, shuffle=False):
        examples = []
        i = 0
        n = 0
        for label, text in dataset:
            i += 1
            tokenized_text = TokenizedText(text, self.tokenizer)
            predicted_label = self._call_model([tokenized_text])[0].argmax().item()
            if predicted_label != label:
                examples.append((label, tokenized_text, True))
            else:
                n += 1
                examples.append((label, tokenized_text, False))
            if num_examples is not None and (n >= num_examples):
                break

        if shuffle:
            random.shuffle(examples)
    
        return examples

    def attack_dataset(self, dataset, num_examples=None, shuffle=False):
        """ 
        Runs an attack on the given dataset and outputs the results to the console and the output file.

        Args:
            dataset: An iterable of (label, text) pairs
            shuffle (:obj:`bool`, optional): Whether to shuffle the data. Defaults to False.
        """
      
        examples = self._get_examples_from_dataset(dataset, num_examples, shuffle)
        results = []

        for label, tokenized_text, was_skipped in examples:
            if was_skipped:
                results.append(SkippedAttackResult(tokenized_text, label))
                continue
            # Start at 1 since we called once to determine that prediction was correct
            self.num_queries = 1
            result = self.attack_one(label, tokenized_text)
            result.num_queries = self.num_queries
            results.append(result)
        
        return results