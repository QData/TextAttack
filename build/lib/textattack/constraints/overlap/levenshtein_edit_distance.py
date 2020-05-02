import editdistance

from textattack.constraints import Constraint

class LevenshteinEditDistance(Constraint):
    """ A constraint on edit distance (Levenshtein Distance).
    """
    
    def __init__(self, max_edit_distance):
        if not isinstance(max_edit_distance, int):
            raise TypeError('max_edit_distance must be an int')
        self.max_edit_distance = max_edit_distance
        
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        edit_distance = editdistance.eval(original_text.text, x_adv.text)
        return edit_distance <= self.max_edit_distance
    
    def extra_repr_keys(self):
        return ['max_edit_distance']
