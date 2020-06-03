import lru
import numpy as np
import torch
import math

from textattack.shared.utils import default_class_repr
from textattack.shared import utils, validators

class GoalFunction:
    """
    Evaluates how well a perturbed tokenized_text object is achieving a specified goal.
    
    Args:
        model: The PyTorch or TensorFlow model used for evaluation.
    """
    def __init__(self, model, use_cache=True):
        validators.validate_model_goal_function_compatibility(self.__class__, model.__class__)
        self.model = model
        self.use_cache = use_cache
        self.num_queries = 0
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
        return self.get_results([tokenized_text], ground_truth_output)[0]

    def get_results(self, tokenized_text_list, ground_truth_output):
        """
        For each tokenized_text object in tokenized_text_list, returns a result 
        consisting of whether or not the goal has been achieved, the output for 
        display purposes, and a score.
        """
        model_outputs = self._call_model(tokenized_text_list)
        results = []
        for tokenized_text, raw_output in zip(tokenized_text_list, model_outputs):
            succeeded = self._is_goal_complete(raw_output, ground_truth_output)
            goal_function_score = self._get_score(raw_output, ground_truth_output)
            displayed_output = self._get_displayed_output(raw_output)
            results.append(
                self._goal_function_result_type()(
                    tokenized_text, displayed_output, 
                    succeeded, goal_function_score)
                )
        return results

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
        outputs need to have a softmax applied. 
        """
        raise NotImplementedError()

    def _call_model_uncached(self, tokenized_text_list, batch_size=utils.config('MODEL_BATCH_SIZE')):
        """ 
        Queries model and returns outputs for a list of TokenizedText 
        objects. 
        """
        if not len(tokenized_text_list):
            return []
        ids = [t.ids for t in tokenized_text_list]
        if hasattr(self.model, 'model'):
            model_device = next(self.model.model.parameters()).device
        else:
            model_device = next(self.model.parameters()).device
        ids = torch.tensor(ids).to(model_device) 
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
        outputs = []
        for batch_i in range(num_batches):
            batch_start = batch_i * batch_size
            batch_stop  = (batch_i + 1) * batch_size
            batch_ids = ids[batch_start:batch_stop]
            batch = [batch_ids[:, x, :] for x in range(num_fields)]
            with torch.no_grad():
                preds = self.model(*batch)
            if isinstance(preds, tuple):
                preds = preds[0]
            outputs.append(preds)
        return self._process_model_outputs(tokenized_text_list, outputs)
    
    def _call_model(self, tokenized_text_list):
        """ Gets predictions for a list of `TokenizedText` objects.
        
            Gets prediction from cache if possible. If prediction is not in the 
            cache, queries model and stores prediction in cache.
        """
        try:
            self.num_queries += len(tokenized_text_list)
        except AttributeError:
            # If some outside class is just using the attack for its `call_model`
            # function, then `self.num_queries` will not have been initialized.
            # In this case, just continue.
            pass
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
        return []
        
    __repr__ = __str__ = default_class_repr
