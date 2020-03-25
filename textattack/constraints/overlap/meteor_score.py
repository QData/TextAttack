import nltk
from textattack.constraints import Constraint

class METEOR(Constraint):
    """ A constraint on METEOR score difference.
    """
    
    def __init__(self, max_meteor):
        if not isinstance(max_meteor, int):
            raise TypeError('max_meteor must be an int')
        self.max_meteor = max_meteor
        
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        meteor = nltk.translate.meteor([original_text], x_adv)  
        return meteor <= self.max_meteor
        
    def extra_repr_keys(self):
        return ['max_meteor']

