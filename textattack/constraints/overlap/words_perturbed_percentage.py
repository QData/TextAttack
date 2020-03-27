from textattack.constraints import Constraint

class WordsPerturbedPercentage(Constraint):
    """ 
    A constraint representing a maximum allowed perturbed words percentage.

    """
    
    def __init__(self, min_percent=None, max_percent=None):
        if min_percent is None and max_percent is None:
            raise ValueError('min and max percent cannot both be None')
        if not (min_percent is None or max_percent is None):
            raise ValueError('must set either min or max percent')
        self.min_percent = min_percent
        self.max_percent = max_percent
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        num_words_diff = len(x_adv.all_words_diff(original_text))
        min_num_words = min(len(x_adv.words), len(original_text.words))
        if self.max_percent:
            max_words_perturbed = round(min_num_words * (self.max_percent / 100))
            return num_words_diff <= max_words_perturbed
        else:
            min_words_perturbed = round(min_num_words * (self.min_percent / 100))
            return num_words_diff >= min_words_perturbed
    def extra_repr_keys(self):
        if self.min_percent is None:
            metric = 'max_percent'
        else:
            metric = 'min_percent'
        return [metric]


