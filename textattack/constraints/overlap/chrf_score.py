import nltk.translate.chrf_score
from textattack.constraints import Constraint

class chrF(Constraint):
    """ A constraint on chrF (n-gram F-score) difference.
    """
    def __init__(self, max_chrf):
        if not isinstance(max_chrf, int):
            raise TypeError('max_chrf must be an int')
        self.max_chrf = max_chrf
    
    def __call__(self, x, x_adv, original_text=None):
        if not original_text:
            return True
        ref = original_text.words
        hyp = x_adv.words
        chrf = nltk.translate.chrf_score.sentence_chrf(ref, hyp)
        return chrf <= self.max_chrf
    
    def extra_repr_keys(self):
        return ['max_chrf']
