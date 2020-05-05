from textattack.constraints import Constraint

class NumberOfWordsSwapped(Constraint):
    """ A constraint representing a maximum number of words swapped. """
    
    def __init__(self, max_swaps=None):
        if max_swaps is None:
            raise ValueError('max_swaps cannot be None')
        self.max_swaps = max_swaps
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        num_words_diff = len(x_adv.all_words_diff(original_text))
        return num_words_diff <= self.max_swaps
        
    def extra_repr_keys(self):
        return ['max_swaps']


