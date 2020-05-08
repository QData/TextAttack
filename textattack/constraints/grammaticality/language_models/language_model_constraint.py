import math
import torch

from textattack.constraints import Constraint

class LanguageModelConstraint(Constraint):
    """ 
        Determines if two sentences have a swapped word that has a similar
            probability according to a language model.
        
        Args:
            max_log_prob_diff (float): the maximum difference in log-probability
                between x and x_adv
    """
    
    def __init__(self, max_log_prob_diff=None):
        if max_log_prob_diff is None:
            raise ValueError('Must set max_log_prob_diff')
        self.max_log_prob_diff = max_log_prob_diff
    
    def get_log_probs_at_index(self, text_list, word_index):
        """ Gets the log-probability of items in `text_list` at index 
            `word_index` according to a language model.
        """
        raise NotImplementedError()
    
    def __call__(self, x, x_adv, original_text=None):
        try:
            i = x_adv.attack_attrs['modified_word_index']
        except AttributeError:
            raise AttributeError('Cannot apply language model constraint without `modified_word_index`')
            
        probs = self.get_log_probs_at_index((x, x_adv), i)
        if len(probs) != 2:
            raise ValueError(f'Error: get_log_probs_at_index returned {len(probs)} values for 2 inputs')
        x_prob, x_adv_prob = probs
        if self.max_log_prob_diff is None:
            x_prob, x_adv_prob = math.log(p1), math.log(p2)
        return abs(x_prob - x_adv_prob) <= self.max_log_prob_diff
    
    def extra_repr_keys(self):
        return ['max_log_prob_diff']
