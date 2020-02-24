import lru
import numpy as np
import torch
import math

from textattack.goal_functions import GoalFunctionResult
from textattack.shared import utils

class GoalFunction:
   
    def __init__(self, model):
        self.model = model
        self.num_queries = 0
        self._call_model_cache = lru.LRU(2**18)

    def should_skip(self, tokenized_text, correct_output):
        self.original_text = tokenized_text
        self.correct_output = correct_output
        model_outputs = self._call_model([tokenized_text])
        return self._is_goal_complete(model_outputs[0])

    def set_original_attrs(self, tokenized_text, correct_output):
        self.original_text = tokenized_text
        self.correct_output = correct_output
        self.num_queries = 1

    def get_results(self, tokenized_text_list):
        model_outputs = self._call_model(tokenized_text_list)
        results = []
        for tokenized_text, raw_output in zip(tokenized_text_list, model_outputs):
            succeeded = self._is_goal_complete(raw_output)
            score = self._get_score(raw_output)
            output = self._get_output(raw_output)
            results.append(GoalFunctionResult(tokenized_text, output, succeeded, score))
        return results

    def _is_goal_complete(self, model_output):
        raise NotImplementedError()

    def _get_score(self, model_output):
        raise NotImplementedError() 

    def _get_output(self, raw_output):
        return raw_output

    def _call_model_uncached(self, tokenized_text_list, batch_size=8):
        """ Queries model and returns predictions for a list of TokenizedText 
            objects. 
        """
        if not len(tokenized_text_list):
            return torch.tensor([])
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
        uncached_list = [text for text in tokenized_text_list if text not in self._call_model_cache]
        scores = self._call_model_uncached(uncached_list)
        for text, score in zip(uncached_list, scores):
            self._call_model_cache[text] = score.cpu()
        final_scores = [self._call_model_cache[text].to(utils.get_device()) for text in tokenized_text_list]
        return torch.stack(final_scores)
