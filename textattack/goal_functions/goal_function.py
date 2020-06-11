import lru
import numpy as np
import torch
import math

from textattack.shared.utils import batch_model_predict, default_class_repr
from textattack.shared import utils, validators

class GoalFunction:
    """
    Evaluates how well a perturbed tokenized_text object is achieving a specified goal.
    
    Args:
        model: The PyTorch or TensorFlow model used for evaluation.
        query_budget: The maximum number of model queries allowed.
    """
    def __init__(self, model, tokenizer=None, use_cache=True, query_budget=float('inf')):
        validators.validate_model_goal_function_compatibility(self.__class__, model.__class__)
        self.model = model
        self.tokenizer = tokenizer
        if not self.tokenizer:
            if hasattr(self.model, 'tokenizer'):
                self.tokenizer = self.model.tokenizer
            else:
                raise NameError('Cannot instantiate goal function without tokenizer')
        if not hasattr(self.tokenizer, 'encode'):
            raise TypeError('Tokenizer must contain `encode()` method')
        self.use_cache = use_cache
        self.num_queries = 0
        self.query_budget = query_budget
        if self.use_cache:
            self._call_model_cache = lru.LRU(utils.config('MODEL_CACHE_SIZE'))
        else:
            self._call_model_cache = None

    def should_skip(self, tokenized_text, ground_truth_output):
        """
        Returns whether or not the goal has already been completed for ``tokenized_text``\,
        due to misprediction by the model.
        """
        model_outputs = self._call_model([tokenized_text])
        return self._is_goal_complete(model_outputs[0], ground_truth_output)

    def get_output(self, tokenized_text):
        """
        Returns output for display based on the result of calling the model.
        """
        return self._get_displayed_output(self._call_model([tokenized_text])[0])
    
    def get_result(self, tokenized_text, ground_truth_output):
        """ 
        A helper method that queries `self.get_results` with a single
        ``TokenizedText`` object.
        """
        results, search_over = self.get_results([tokenized_text], ground_truth_output)
        result = results[0] if len(results) else None
        return result, search_over

    def get_results(self, tokenized_text_list, ground_truth_output):
        """
        For each tokenized_text object in tokenized_text_list, returns a result 
        consisting of whether or not the goal has been achieved, the output for 
        display purposes, and a score. Additionally returns whether the search
        is over due to the query budget.
        """
        results = []
        if self.query_budget < float('inf'):
            queries_left = self.query_budget - self.num_queries
            tokenized_text_list = tokenized_text_list[:queries_left]
        self.num_queries += len(tokenized_text_list)
        model_outputs = self._call_model(tokenized_text_list)
        for tokenized_text, raw_output in zip(tokenized_text_list, model_outputs):
            displayed_output = self._get_displayed_output(raw_output)
            succeeded = self._is_goal_complete(raw_output, ground_truth_output)
            goal_function_score = self._get_score(raw_output, ground_truth_output)
            results.append(
                self._goal_function_result_type()(
                    tokenized_text, displayed_output, 
                    succeeded, goal_function_score)
                )
        return results, self.num_queries == self.query_budget

    def _is_goal_complete(self, model_output, ground_truth_output):
        raise NotImplementedError()

    def _get_score(self, model_output, ground_truth_output):
        raise NotImplementedError() 

    def _get_displayed_output(self, raw_output):
        return raw_output
    
    def _goal_function_result_type(self):
        """ 
        Returns the class of this goal function's results. 
        """
        raise NotImplementedError()
    
    def _process_model_outputs(self, inputs, outputs):
        """ 
        Processes and validates a list of model outputs. 
        
        This is a task-dependent operation. For example, classification 
        outputs need to make sure they have a softmax applied. 
        """
        raise NotImplementedError()

    def _call_model_uncached(self, tokenized_text_list):
        """ 
        Queries model and returns outputs for a list of TokenizedText 
        objects. 
        """
        if not len(tokenized_text_list):
            return []
        ids = [t.ids for t in tokenized_text_list]
        
        with torch.no_grad():
            outputs = batch_model_predict(self.model, ids)
        
        return self._process_model_outputs(tokenized_text_list, outputs)
    
    def _call_model(self, tokenized_text_list):
        """ Gets predictions for a list of `TokenizedText` objects.
        
            Gets prediction from cache if possible. If prediction is not in the 
            cache, queries model and stores prediction in cache.
        """
        if not self.use_cache:
            return self._call_model_uncached(tokenized_text_list)
        else:
            uncached_list = []
            for text in tokenized_text_list:
                if text in self._call_model_cache:
                    # Re-write value in cache. This moves the key to the top of the
                    # LRU cache and prevents the unlikely event that the text
                    # is overwritten when we store the inputs from `uncached_list`.
                    self._call_model_cache[text] = self._call_model_cache[text]
                else:
                    uncached_list.append(text)
            uncached_list = [text for text in tokenized_text_list if text not in self._call_model_cache]
            outputs = self._call_model_uncached(uncached_list)
            for text, output in zip(uncached_list, outputs):
                self._call_model_cache[text] = output
            all_outputs = [self._call_model_cache[text] for text in tokenized_text_list]
            return all_outputs

    def extra_repr_keys(self): 
        if self.query_budget < float('inf'):
            return ['query_budget']
        return []
        
    __repr__ = __str__ = default_class_repr
